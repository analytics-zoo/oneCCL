#include <fcntl.h>
#include <unistd.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/socket.h>

#include <drm/drm.h>

#include <mpi.h>

#include <vector>
#include <sstream>
#include <iostream>

#include <ext/intel/esimd.hpp>
//#include <level_zero/ze_api.h>

#include "oneapi/ccl.hpp"
#include "common/global/global.hpp"
#include "common/utils/exchange_utils.hpp"

#include "dg2_allreduce.hpp"

using namespace std;
using namespace sycl;

static sycl::queue q;

static int world_size = 0;
static int world_rank = 0;
static void *host_bufs[DG2_NUM];       /* host shared buf */
static void *peer_bufs[DG2_NUM];       /* shared buf on peer side */

#define FLAG_SIZE (32)                 /* FIXME: 256 bits */
#define SOCK_PATH "/tmp/ipc_sock"


uint16_t pattern_counter = 0xa770;

typedef uint32_t pattern_t;
using message_t = sycl::vec<uint32_t, 4>;


#define SG_SZ      (16)                        /* Arc770: Subgroup Sizes Supported: 8;16;32, while 8 threads per EU */
#define LS_SZ      (sizeof(message_t))         /* load/store byte size per work-item */

#define __LscLoadUnCached(var, addr)   \
    __asm__ __volatile__("lsc_load.ugm.uc.uc   (M1, 16)  %0:d64  flat[%1]:a64" : "=rw"(var) : "rw"(addr) : "memory")
#define __LscLoadCached(var, addr)   \
    __asm__ __volatile__("lsc_load.ugm.ca.ca   (M1, 16)  %0:d64  flat[%1]:a64" : "=rw"(var) : "rw"(addr) : "memory")
#define __LscLoadUnCachedVec(var, addr)   \
    __asm__ __volatile__("lsc_load.ugm.uc.uc   (M1, 16)  %0:d32x4  flat[%1]:a64" : "=rw"(reinterpret_cast<typename message_t::vector_t &>(var)) : "rw"(addr) : "memory")
#define __LscLoadCachedVec(var, addr)   \
    __asm__ __volatile__("lsc_load.ugm.ca.ca   (M1, 16)  %0:d32x4  flat[%1]:a64" : "=rw"(reinterpret_cast<typename message_t::vector_t &>(var)) : "rw"(addr) : "memory")

#define __LscStoreUnCached(addr, var)  \
    __asm__ __volatile__("lsc_store.ugm.uc.uc  (M1, 16)  flat[%0]:a64  %1:d64" : : "rw"(addr), "rw"(var) : "memory")
#define __LscStoreCached(addr, var)  \
    __asm__ __volatile__("lsc_store.ugm.ca.ca  (M1, 16)  flat[%0]:a64  %1:d64" : : "rw"(addr), "rw"(var) : "memory")
#define __LscStoreUnCachedVec(addr, var)  \
    __asm__ __volatile__("lsc_store.ugm.uc.uc  (M1, 16)  flat[%0]:a64  %1:d32x4" : : "rw"(addr), "rw"(reinterpret_cast<typename message_t::vector_t &>(var)) : "memory")
#define __LscStoreCachedVec(addr, var)  \
    __asm__ __volatile__("lsc_store.ugm.ca.ca  (M1, 16)  flat[%0]:a64  %1:d32x4" : : "rw"(addr), "rw"(reinterpret_cast<typename message_t::vector_t &>(var)) : "memory")

#define LscLoadCached     __LscLoadCachedVec
#define LscLoadUnCached   __LscLoadUnCachedVec
#define LscStoreCached    __LscStoreCachedVec
#define LscStoreUnCached  __LscStoreUnCachedVec


inline int get_fd_from_handle(const ze_ipc_mem_handle_t& handle)
{
    return *(reinterpret_cast<const int*>(handle.data));
}

inline ze_ipc_mem_handle_t get_handle_from_fd(int fd)
{
    ze_ipc_mem_handle_t handle{};
    *(reinterpret_cast<int*>(handle.data)) = fd;
    return handle;
}

static int srv_sock(char *sock_path)
{
    int fd;
    struct sockaddr_un sk;

    sk.sun_family = AF_UNIX;
    strncpy(sk.sun_path, sock_path, sizeof(sk.sun_path)-1);

    unlink(sock_path);
    fd = socket(PF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }

    mode_t prev_umask = umask(0);
    if (fchmod(fd, 0666) < 0) {
        close(fd);
        return -1;
    }
    umask(prev_umask);

    if (bind(fd, (struct sockaddr *)&sk, sizeof(sk)) == -1) {
        close(fd);
        return -1;
    }

    if (fcntl(fd, F_SETFL, O_NONBLOCK) == -1) {
        close(fd);
        return -1;
    }

    if (listen(fd, 5) == -1) {
        close(fd);
        return -1;
    }

    return fd;
}

static int cli_sock(char *sock_path)
{
    int fd;
    struct sockaddr_un sk;

    sk.sun_family = AF_UNIX;
    strncpy(sk.sun_path, sock_path, sizeof(sk.sun_path)-1);

    fd = socket(PF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        fprintf(stderr, "Failed to create socket(%d), sock path: %s\n", errno, sock_path);
        return -1;
    }

    if (connect(fd, (struct sockaddr *)&sk, sizeof(sk)) == -1) {
        //fprintf(stderr, "Failed to connect socket(%d)\n", errno);
        close(fd);
        return -1;
    }

    return fd;
}

static void *thread_func(void *arg)
{
    fd_set fds;
    int count = 0;
    char sock_path[64];
    int peer_buf_fd = 0;
    int rank = *(int *)arg;

    snprintf(sock_path, sizeof(sock_path), "%s-%d_%d", SOCK_PATH, rank, 0xa770);
    int srv_fd = srv_sock(sock_path);

    //std::cout << "-----> srv_fd of " << sock_path << " : " << srv_fd << "\n";

    auto sycl_context = q.get_context();
    auto sycl_device = q.get_device();
    ze_context_handle_t ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_context);
    ze_device_handle_t  ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);

    FD_ZERO(&fds);
    FD_SET(srv_fd, &fds);
    while (++count < world_size) {
        int ret = select(srv_fd + 1, &fds, NULL, NULL, NULL);
        switch (ret) {
        case 1:
            {
                int peer_rank;
                void *peer_buf;

                int conn_fd = accept(srv_fd, NULL, 0);
                ccl::utils::recvmsg_fd(conn_fd, &peer_buf_fd, &peer_rank, sizeof(peer_rank));

                ze_ipc_mem_handle_t ipc_handle_peer_buf = get_handle_from_fd(peer_buf_fd);
                zeMemOpenIpcHandle(ze_context, ze_device, ipc_handle_peer_buf, ZE_IPC_MEMORY_FLAG_BIAS_CACHED /* cached allocation */, &peer_buf);

                peer_bufs[peer_rank] = peer_buf;
                //printf("<------------- rank: %d, peer_bufs[%d]: %p\n", world_rank, peer_rank, peer_bufs[peer_rank]);

                if (conn_fd > 0) close(conn_fd);

                break;
            }
        case 0:
        case -1:
            std::cout << "srv_fd select() failed" << "\n";
            break;
        default:
            break;
        }
    }

    if (srv_fd > 0) {
        close(srv_fd);
        unlink(sock_path);
    }

    return nullptr;
}

void create_shared_buf(void *send_buf, void *recv_buf, size_t byte_count)
{
    printf("-----> current rank: %d, world size: %d, byte_count: %lu\n", world_rank, world_size, byte_count);

    pthread_t tid;
    char sock_path[64];
    void *host_buf = nullptr;

    auto sycl_context = q.get_context();
    ze_context_handle_t ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_context);

    /* thread to accept connection request from peers */
    pthread_create(&tid, nullptr, thread_func, &world_rank);

    size_t buf_size = LL256_BUF_SIZE;
    host_buf = sycl::aligned_alloc_device(getpagesize(), buf_size, q);

    host_bufs[world_rank] = host_buf;

    //printf("-------------> rank: %d, host_bufs[%d]: %p\n", world_rank, world_rank, host_bufs[world_rank]);

    for (int i = 0; i < world_size; i++) {
        if (i != world_rank) {
            int cli_fd = -1;

            ze_ipc_mem_handle_t ipc_handle_host_buf;
            zeMemGetIpcHandle(ze_context, host_buf, &ipc_handle_host_buf);
            int host_buf_fd = get_fd_from_handle(ipc_handle_host_buf);

            snprintf(sock_path, sizeof(sock_path), "%s-%d_%d", SOCK_PATH, i, 0xa770);
            while (cli_fd < 0) cli_fd = cli_sock(sock_path);
            //std::cout << "<----- cli_fd of " << sock_path << " : " << cli_fd << "\n";

            ccl::utils::sendmsg_fd(cli_fd, host_buf_fd, &world_rank, sizeof(world_rank));

            if (cli_fd > 0) close(cli_fd);
        }
    }

    pthread_join(tid, nullptr);
}

void dg2_init(ccl_coll_param &param)
{
    size_t byte_count = param.count * param.dtype.size();

    ccl_stream *stream = param.stream;
    q = stream->get_native_stream();
    // init recv_buf in advance to warm up GPU
    if (param.send_bufs[0] != param.recv_bufs[0])
        q.memcpy(param.recv_bufs[0], param.send_bufs[0], byte_count);

    /* init already */
    if (world_size != 0)
        return;

    ccl_comm *comm = param.comm;
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_node_comm().get()->get_atl_comm();
    world_size = atl_comm->get_size();
    world_rank = atl_comm->get_rank();

    create_shared_buf(param.send_bufs[0], param.recv_bufs[0], byte_count);
}


template <typename T>
static inline message_t _sum(message_t dst, message_t src)
{
    using math_t = sycl::vec<T, sizeof(message_t) / sizeof(T)>;
    return sycl::bit_cast<message_t>(sycl::bit_cast<math_t>(dst) + sycl::bit_cast<math_t>(src));
}

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
static inline message_t sum(message_t dst, message_t src, const ccl_datatype& dtype)
{
    message_t data;

    switch (dtype.idx()) {
    case ccl::datatype::float16:
        data = _sum<sycl::half>(dst, src);
        break;

    case ccl::datatype::float32:
        data = _sum<float>(dst, src);
        break;

    case ccl::datatype::int32:
        data = _sum<int32_t>(dst, src);
        break;

    default:
        /* following code will hurt performance */
        //sycl::ext::oneapi::experimental::printf("Unknow dtype!\n");
        break;
    }

    return data;
}

static inline void sync_data(char *src, message_t &data, int lid, pattern_t pattern)
{
    size_t sz = sizeof(message_t);
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();

    do {
        LscLoadUnCached(data, src + lid * sz);
    } while (sycl::any_of_group(sg, ((lid ==  3) && (data[3] != pattern)) ||
                                    ((lid ==  7) && (data[3] != pattern)) ||
                                    ((lid == 11) && (data[3] != pattern)) ||
                                    ((lid == 15) && (data[3] != pattern))));
}

static inline void shuffle_data(message_t &data)
{
    __asm__ __volatile__("mov (M1, 1) %0(1, 7)<1> %0(6, 3)<0;1,0>\n"
                         "mov (M1, 1) %0(3, 7)<1> %0(6, 7)<0;1,0>\n"
                         "mov (M1, 1) %0(5, 7)<1> %0(7, 3)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(data))
                         : );
}

static inline void insert_pattern(message_t &data, pattern_t pattern)
{
    __asm__ __volatile__("mov (M1, 1) %0(6, 3)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(6, 7)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(7, 3)<1> %1(0, 0)<0;1,0>\n"
                         "mov (M1, 1) %0(7, 7)<1> %1(0, 0)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(data))
                         : "rw"(pattern));
}

static inline void restore_data(message_t &data)
{
    __asm__ __volatile__("mov (M1, 1) %0(6, 3)<1> %0(1, 7)<0;1,0>\n"
                         "mov (M1, 1) %0(6, 7)<1> %0(3, 7)<0;1,0>\n"
                         "mov (M1, 1) %0(7, 3)<1> %0(5, 7)<0;1,0>\n"
                         : "+rw"(reinterpret_cast<typename message_t::vector_t &>(data))
                         : );
}
#endif

static inline void send(char *next, char *src, int lid, int req_workitems,
                        const ccl_datatype& dtype, int rank, pattern_t pattern)
{
    #if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    message_t data;
    int sz = sizeof(data);

    LscLoadCached(data, src + lid * sz);

    shuffle_data(data);
    insert_pattern(data, pattern);

    LscStoreUnCached(next + lid * sz, data);
    #endif
}

static inline void recv_reduce_send(char *dst, char *next, char *src, int lid, int req_workitems,
                                    const ccl_datatype& dtype, int rank, pattern_t pattern)
{
    #if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    message_t data;
    int sz = sizeof(data);
    message_t *dst_buf = (message_t *)dst;

    sync_data(src, data, lid, pattern);
    restore_data(data);

    data = sum(dst_buf[lid], data, dtype);

    shuffle_data(data);
    insert_pattern(data, pattern);
    LscStoreUnCached(next + lid * sz, data);
    #endif
}

static inline void recv_reduce_copy_send(char *dst, char *next, char *src, int lid, int req_workitems,
                                         const ccl_datatype& dtype, int rank, pattern_t pattern)
{
    #if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    message_t data;
    int sz = sizeof(data);
    message_t *dst_buf = (message_t *)dst;

    sync_data(src, data, lid, pattern);
    restore_data(data);

    data = sum(dst_buf[lid], data, dtype);
    if (lid < req_workitems)
        LscStoreUnCached(dst + lid * sz, data);

    shuffle_data(data);
    insert_pattern(data, pattern);
    LscStoreUnCached(next + lid * sz, data);
    #endif
}

static inline void recv_copy_send(char *dst, char *next, char *src, int lid, int req_workitems,
                                  const ccl_datatype& dtype, int rank, pattern_t pattern)
{
    #if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    message_t data;
    int sz = sizeof(data);

    sync_data(src, data, lid, pattern);
    LscStoreUnCached(next + lid * sz, data);

    restore_data(data);

    if (lid < req_workitems)
        LscStoreUnCached(dst + lid * sz, data);
    #endif
}

static inline void recv(char *dst, char *src, int lid, int req_workitems,
                        const ccl_datatype& dtype, int rank, pattern_t pattern)
{
    #if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    message_t data;
    int sz = sizeof(data);

    /* copy reduced data from peer */
    sync_data(src, data, lid, pattern);

    restore_data(data);

    if (lid < req_workitems)
        LscStoreUnCached(dst + lid * sz, data);
    #endif
}


ccl::event dg2_ll256_allreduce(const void *src, void *dst, size_t count,
                               const ccl_datatype &dtype, ccl::reduction reduction, ccl_comm *comm)
{
    ccl::event ret;

    //std::cout << "enter " << __func__ << ", rank: " << world_rank <<  ", count: " << count << std::endl;

    size_t dt_sz = dtype.size();
    char *recv_buf = static_cast<char *>(dst);
    char *send_buf = static_cast<char *>(const_cast<void*>(src));

    /*
     * Intel(R) Arc(TM) A770 Graphics:
     *   Number Of Slices:                       1
     *   Number Of Subslices Per Slice:          32
     *   Number Of EU Per Subslice:              16
     *   Number Of Threads Per EU:               8
     *   Total EU Count:                         512
     *   Physical EU SIMD Width:                 8
     */

    /* 64-byte load/store granularity to HBM, Maximum 128-byte payload can be used by EU store */
    /* Arc770: Subgroup Sizes Supported: 8;16;32, while 8 threads per EU */
    size_t sg_sz = SG_SZ;

    size_t l_sz = 1 * sg_sz;
    size_t g_sz = 512 * l_sz;

    /* To avoid pattern not changed when "iters" is 1 */
    pattern_t pattern_prefix = ++pattern_counter << 16;

    q.submit([&](auto& h) {
        using namespace sycl::ext::intel::experimental::esimd;

        int local_world_rank = world_rank;
        int local_world_size = world_size;

        int next_rank = (local_world_rank + 1) % local_world_size;
        char *local_host_buf = (char *)host_bufs[local_world_rank];

        char *local_peer_bufs[DG2_NUM];
        for (int i = 0; i < world_size; i++)
            local_peer_bufs[i] = (char *)peer_bufs[i];

        /*
         * In a single subgroup:
         *   a> 1 dedicated work-item to manage a LS_SZ-byte pattern.
         *   b> other work-items to process data, and each of them handle a LS_SZ-byte data.
         */
        auto default_subgroup_capacity = sg_sz * LS_SZ;  /* bytes: data and pattern  processed by 1 subgroup */
        auto default_workgroup_capacity = l_sz * LS_SZ;  /* bytes: data and patterns processed by 1 workgroup */
        //auto default_total_capacity = g_sz * LS_SZ;      /* bytes: data and patterns processed by all workgroups in 1 iteration */

        /* In a single workgroup, the available work-items to process data, excluding work-items for patterns */
        auto workgroup_available_items = l_sz - (l_sz / sg_sz);
        auto total_available_items = (g_sz / l_sz) * workgroup_available_items;

        auto subgroup_capacity = LS_SZ * (sg_sz - 1);                  /* bytes: data processed by 1 subgroup */
        auto workgroup_capacity = LS_SZ * workgroup_available_items;   /* bytes: data processed by 1 workgroup */
        auto total_capacity = (g_sz / l_sz) * workgroup_capacity;      /* bytes: data processed by all workgroups in 1 iteration */

        /* div up */
        int iters = (count * dt_sz + (local_world_size * total_available_items * LS_SZ - 1)) / (local_world_size * total_available_items * LS_SZ);

        //sycl::ext::oneapi::experimental::printf("------> rank: %d, group num: %ld, loop count: %zu\n", local_world_rank, g_sz / l_sz, iters);

        h.parallel_for(sycl::nd_range<1>(g_sz, l_sz), [=] (sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SZ)]] {
            int idx = 0;
            size_t offset = 0;
            size_t offset_with_pattern = 0;

            auto group_id = item.get_group_linear_id();
            auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
            auto sg_id = sg.get_group_id()[0];
            auto sg_lid = sg.get_local_id()[0];

            for (int i = 0; i < iters; i++) {
                pattern_t pattern = pattern_prefix + i;

                auto base = local_world_size * (i        * total_capacity  +
                                                group_id * workgroup_capacity  +
                                                sg_id    * subgroup_capacity);
                auto base_with_pattern = local_world_size * (/* i        * default_total_capacity  + */
                                                             group_id * default_workgroup_capacity  +
                                                             sg_id    * default_subgroup_capacity);

                auto finished = i * total_capacity * local_world_size;   /* bytes */
                auto unreduced = count * dt_sz - finished;               /* bytes */

                auto req_workitems = sg_sz - 1;                /* required work-items exclude 1 work-item for pattern */
                auto chunk_sz = req_workitems * LS_SZ;         /* LS_SZ bytes per work-item */
                auto chunk_with_pattern = sg_sz * LS_SZ;       /* aligned to 256B */

		auto work_left = unreduced - sg_id * local_world_size * chunk_sz;

                if (work_left > 0) {
                    // step 1: push data to next GPU
                    {
                        offset = base + local_world_rank * chunk_sz;
                        offset_with_pattern = base_with_pattern + local_world_rank * chunk_with_pattern;

                        char *next = local_peer_bufs[next_rank];

                        send(next + offset_with_pattern, send_buf + offset, sg_lid, req_workitems, dtype, local_world_rank, pattern);
                    }

                    // step 2: reduce and copy to next GPU
                    for (int j = 2; j < local_world_size; j++) {
                        idx = (local_world_rank + local_world_size + 1 - j) % local_world_size;
                        offset = base + idx * chunk_sz;
                        offset_with_pattern = base_with_pattern + idx * chunk_with_pattern;

                        char *src = local_host_buf;
                        char *next = local_peer_bufs[next_rank];

                        recv_reduce_send(recv_buf + offset, next + offset_with_pattern, src + offset_with_pattern,
                                         sg_lid, req_workitems, dtype, local_world_rank, pattern);
                    }

                    // step 3: reduce this buffer and data, which will produce the final
                    // result that we store in this data and push to the next GPU
                    {
                        idx = (local_world_rank + 1) % local_world_size;
                        offset = base + idx * chunk_sz;
                        offset_with_pattern = base_with_pattern + idx * chunk_with_pattern;

                        char *src = local_host_buf;
                        char *next = local_peer_bufs[next_rank];

                        recv_reduce_copy_send(recv_buf + offset, next + GATHER_BUF_OFFSET + offset_with_pattern, src + offset_with_pattern,
                                              sg_lid, req_workitems, dtype, local_world_rank, pattern);
                    }

                    // step 4: copy to next GPU
                    for (int j = 1; j < local_world_size - 1; ++j) {
                        idx = (local_world_rank + local_world_size + 1 - j) % local_world_size;
                        offset = base + idx * chunk_sz;
                        offset_with_pattern = GATHER_BUF_OFFSET + base_with_pattern + idx * chunk_with_pattern;

                        char *src = local_host_buf;
                        char *next = local_peer_bufs[next_rank];

                        recv_copy_send(recv_buf + offset, next + offset_with_pattern, src + offset_with_pattern,
                                       sg_lid, req_workitems, dtype, local_world_rank, pattern);
                    }

                    // step 5: Make final copy from buffer to dest
                    {
                        idx = (local_world_rank + 2) % local_world_size;
                        offset = base + idx * chunk_sz;
                        offset_with_pattern = GATHER_BUF_OFFSET + base_with_pattern + idx * chunk_with_pattern;

                        char *src = local_host_buf;

                        recv(recv_buf + offset, src + offset_with_pattern, sg_lid, req_workitems, dtype, local_world_rank, pattern);
                    }
                }
            }
        });
    });

    return ret;
}

ccl::event dg2_allreduce(const void *src, void *dst, size_t count,
                         const ccl_datatype &dtype, ccl::reduction reduction, ccl_comm *comm)
{
    ccl::event ret;

    dg2_ll256_allreduce(src, dst, count, dtype, reduction, comm);

    return ret;
}

void dg2_clear()
{
    for (int i = 0; i < world_size; i++) {
        if (i == world_rank)
            continue;

        auto host_buf = host_bufs[world_rank];
        if (host_buf)
            sycl::free(host_buf, q);
    }
}
