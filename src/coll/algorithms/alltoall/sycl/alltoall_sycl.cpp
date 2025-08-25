/*
 Copyright 2016-2020 Intel Corporation
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
#include "coll/algorithms/alltoall/sycl/alltoall_sycl.hpp"
#include "coll/algorithms/allreduce/sycl/allreduce_ring_ll256.hpp"
#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_ll256.hpp"
#endif // CCL_ENABLE_SYCL

#define ALLTOALL_LOOP_BUF_OFFSET (3 * 16 * 1024 * 1024 / 2)

using namespace std;
using namespace sycl;

#define SG_SZ (16) /* Arc770: Subgroup Sizes Supported: 8;16;32, while 8 threads per EU */
#define LS_SZ (sizeof(message_t)) /* load/store byte size per work-item */

static inline void alltoall_ll256_send(char *src, char *dst, bool load, pattern_t pattern,int lid,size_t left_size) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    message_t data;
    int sz = sizeof(data);

      if ( (lid * sz < left_size) ) {
          LscLoadCached(data, src);
      }

    shuffle_data(data);

    insert_pattern(data, pattern);

    LscStoreUnCached(dst, data);
#endif
}

static inline void alltoall_ll256_recv(char *recvbuf,
                              char *tmpbuf,
                              sycl::sub_group &sg,
                              int lid,
                              int req_workitems,
                              pattern_t pattern,
                              size_t left_size) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    message_t data;
    int sz = sizeof(data);

    /* check if data arrived in src */
    sync_data(tmpbuf, data, sg, lid, pattern);

    restore_data(data);

    if ((lid < req_workitems) && (lid * sz < left_size) ) {
        LscStoreUnCached(recvbuf, data);
    }

#endif
}

namespace ccl {
namespace v1 {

sycl::event arc_ll256_alltoall(const void *src,
                                void *dst,
                                size_t count,
                                ccl::datatype dtype,
                                ccl_comm *comm,
                                ccl_stream *global_stream) {
    sycl::event sycl_e;

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    const int comm_size = node_comm->size();
    const int comm_rank = node_comm->rank();

    //std::cout << "enter " << __func__ << ", rank: " << comm_rank <<  ", count: " << count << std::endl;

    sycl::queue q = global_stream->get_native_stream();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t dt_sz = ccl_dtype.size();
    char *recv_buf = static_cast<char *>(dst);
    char *send_buf = static_cast<char *>(const_cast<void *>(src));

    if (send_buf != recv_buf)
        sycl_e = q.memcpy(recv_buf, send_buf, dt_sz * count * comm_size);

    /*
     * Intel(R) Arc(TM) A770 Graphics:
     *   Number Of Slices:                       1
     *   Number Of Subslices Per Slice:          32
     *   Number Of EU Per Subslice:              16
     *   Number Of Threads Per EU:               8
     *   Total EU Count:                         512
     *   Physical EU SIMD Width:                 8
     *   GRF size                                32B
     */

    /*
     * B580 Graphics (B-series):
     *   Number Of Slices:                       5
     *   Number Of Subslices Per Slice:          4
     *   Number Of EU Per Subslice:              8
     *   Number Of Threads Per EU:               8
     *   Total EU Count:                         1280
     *   GRF size                                64B
     */

    /* 64-byte load/store granularity to HBM, Maximum 128-byte payload can be used by EU store */
    /* Arc770: Subgroup Sizes Supported: 8;16;32, while 8 threads per EU */
    size_t sg_sz = SG_SZ;

    // query total number of hardware threads
    ze_device_handle_t ze_dev =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    ssize_t dev_id{ ccl::utils::invalid_device_id };
    if (!ccl::ze::get_device_global_id(ze_dev, &dev_id)) {
        CCL_THROW("unable to get global id for device\n");
    }

    size_t l_sz = 1 * sg_sz;
    int ngroups = ccl::global_data::get().ze_data->devices[dev_id].total_threads;
    size_t g_sz = ngroups * l_sz;
    auto max_threads = ngroups * 16;

    /* To avoid pattern not changed when "iters" is 1 */
    pattern_t pattern_prefix = ++pattern_counter << 16;

    sycl_e = q.submit([&](auto &h) {
        //using namespace sycl::ext::intel::experimental::esimd;

        int local_world_rank = comm_rank;
        int local_world_size = comm_size;

        int next_rank = (local_world_rank + 1) % local_world_size;

        char *local_peer_bufs[ARC_NUM];

        // use large kernel persistent buffers
        for (int i = 0; i < local_world_size; i++) {
            local_peer_bufs[i] = (char *)get_remote_node_tmp_buf(0, comm)[i];
        }
        //char *local_tmp_buf = local_peer_bufs[local_world_rank];
        char *local_tmp_buf = (char *)get_tmp_buf(0, comm);

        /*
         * In a single subgroup:
         *   a> 1 dedicated work-item to manage a LS_SZ-byte pattern.
         *   b> other work-items to process data, and each of them handle a LS_SZ-byte data.
         */
        auto default_subgroup_capacity =
            sg_sz * LS_SZ; /* bytes: data and pattern  processed by 1 subgroup */
        auto default_workgroup_capacity =
            l_sz * LS_SZ; /* bytes: data and patterns processed by 1 workgroup */
        //auto default_total_capacity = g_sz * LS_SZ;      /* bytes: data and patterns processed by all workgroups in 1 iteration */

        /* In a single workgroup, the available work-items to process data, excluding work-items for patterns */
        auto workgroup_available_items = l_sz - (l_sz / sg_sz);

        auto subgroup_capacity = LS_SZ * (sg_sz - 1); /* bytes: data processed by 1 subgroup */
        // bytes: data processed by 1 workgroup
        auto workgroup_capacity = LS_SZ * workgroup_available_items;

        // calculate how many total threads to dispatch
        size_t bytes_per_rank = count * dt_sz;//(count * dt_sz + local_world_size - 1) / local_world_size;
        ngroups = (bytes_per_rank + subgroup_capacity - 1) / subgroup_capacity;
        g_sz = ngroups * l_sz;
        if (g_sz > max_threads) {
            ngroups = max_threads / l_sz;
            g_sz = ngroups * l_sz;
        }

        auto total_available_items = ngroups * workgroup_available_items;
        // bytes: data processed by all workgroups in 1 iteration
        auto total_capacity = ngroups * workgroup_capacity;

        /* div up */
        int iters = (count * dt_sz + ( total_available_items * LS_SZ - 1)) /
                    ( total_available_items * LS_SZ);

        h.parallel_for(
            sycl::nd_range<1>(g_sz, l_sz),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SZ)]] {
                int idx = 0;
                size_t offset = 0;
                size_t offset_with_pattern = 0;

                auto group_id = item.get_group_linear_id();
                //auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
                auto sg = item.get_sub_group();
                auto sg_id = sg.get_group_id()[0];
                auto sg_lid = sg.get_local_id()[0];

                for (int i = 0; i < iters; i++) {

                    // base offsets of the current subgroup
                    auto base =
                         (i * total_capacity + group_id * workgroup_capacity +
                                            sg_id * subgroup_capacity);
                    auto base_with_pattern =
                        (group_id * default_workgroup_capacity + sg_id * default_subgroup_capacity);

                    auto finished = i * total_capacity ; /* bytes */
                    auto unreduced = count * dt_sz - finished; /* bytes */

                    // required work-items exclude 1 work-item for pattern
                    auto req_workitems = sg_sz - 1;
                    // LS_SZ bytes per work-item
                    auto chunk_sz = req_workitems * LS_SZ;
                    // aligned to 256B
                    auto chunk_with_pattern = sg_sz * LS_SZ;

                    /* items will be assigned to each rank */
                    auto per_rank_items =
                        (unreduced + ( LS_SZ - 1)) / ( LS_SZ);
                    auto req_workgroups = (per_rank_items + (workgroup_available_items - 1)) /
                                          workgroup_available_items;
                    auto req_subgroups = 0;

                    if (req_workgroups >= ngroups) {
                        req_workgroups = ngroups;
                    }
                    else {
                        if (group_id == (req_workgroups - 1)) {
                            req_subgroups = (per_rank_items + (sg_sz - 1)) / (sg_sz - 1);

                            /* (req_subgroups % (l_sz/sg_sz) - 1) equals to the final subgroup id in a workgroup */
                            /* Note:  req_subgroups % (l_sz/sg_sz) might be 0 */
                            if (((req_subgroups % (l_sz / sg_sz)) == 0) ||
                                (sg_id == (req_subgroups % (l_sz / sg_sz) - 1))) {
                                if ((per_rank_items % (sg_sz - 1)) != 0) {
                                    /* FIXME: */
                                    req_workitems = per_rank_items % (sg_sz - 1);
                                    // LS_SZ bytes per work-item
                                    chunk_sz = req_workitems * LS_SZ;
                                }
                            }
                        }
                    }

/*
//                if ((group_id < req_workgroups) && (sg_lid < req_workitems)) {
                  if (i == 0) {
                    sycl::ext::oneapi::experimental::printf(
                        "rank: %d, i: %d, base: %d, grp id: %d, sg_lid: %d, unreduced: %d, req_workgroups: %d, req_items: %d, chunk_sz: %d, req_subgroups: %d, per_rank_items: %d,ngroups:%d\n",
                         local_world_rank, i, base, group_id, sg_lid, unreduced, req_workgroups, req_workitems, chunk_sz, req_subgroups, per_rank_items,ngroups);
                }
*/

                    if (group_id < req_workgroups) {

                        for (int k = 1; k < local_world_size; k++) {
			  pattern_t pattern = pattern_prefix + (i<<8) + k;

			int next_rank = local_world_rank ^ k;
			char *next = local_peer_bufs[next_rank];


                        // step 1: send data to dest GPU
                        {
			    offset = base + next_rank*count*dt_sz ;
                            offset_with_pattern =
                                base_with_pattern;

			    if (i % 2 != 0)
				offset_with_pattern += ALLTOALL_LOOP_BUF_OFFSET;

                                offset_with_pattern += (k-1) * ALLTOALL_LOOP_BUF_OFFSET /local_world_size ;

                            size_t left_size = count * dt_sz - base;
                            alltoall_ll256_send(send_buf + offset + sg_lid * LS_SZ,
                                       next + offset_with_pattern + sg_lid * LS_SZ,
                                       sg_lid * LS_SZ < left_size,
                                       pattern,sg_lid,left_size);
                        }


                        // step 2:recv data from dest
                        {
                            offset = base + next_rank*count*dt_sz;
                            offset_with_pattern = base_with_pattern;

                            if (i % 2 != 0)
                                offset_with_pattern += ALLTOALL_LOOP_BUF_OFFSET;

                                offset_with_pattern += (k-1) * ALLTOALL_LOOP_BUF_OFFSET /local_world_size;

			    size_t left_size = count * dt_sz - base;
                            alltoall_ll256_recv(recv_buf + offset + sg_lid * LS_SZ,
                                       local_tmp_buf  + offset_with_pattern + sg_lid * LS_SZ,
                                       sg,
                                       sg_lid,
                                       req_workitems,
                                       pattern,left_size);
                        }
			}
                    }
                }
            });
    });

    return sycl_e;
}

ccl::event arc_alltoall(const void *src,
                         void *dst,
                         size_t count,
                         ccl::datatype dtype,
                         ccl_comm *comm,
                         ccl_stream *global_stream) {
    coll_init(comm, global_stream);

    auto e = arc_ll256_alltoall(src, dst, count, dtype, comm, global_stream);

    return ccl::event::create_from_native(e);
}

ccl::event alltoall_sycl_single_node(sycl::queue& q,
                                     const void* send_buf,
                                     void* recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     ccl_comm* comm,
                                     ccl_stream* global_stream,
                                     const vector_class<event>& deps,
                                     bool& done) {
    ccl::event e;
    done = true;

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    //
    //     const bool is_single_tile = comm->get_pair_comm()->size() == 1;
    const bool has_all_vertices_connected = comm->get_topo_manager().has_all_vertices_connected();
    LOG_DEBUG("|CCL_SYCL| has_all_vertices_connected", has_all_vertices_connected);
    //
    uint32_t world = comm->get_node_comm()->size();
    int rank = comm->get_node_comm()->rank();

    if (world == 1) {
        sycl::event sycl_e;
        std::vector<sycl::event> dep_events = get_sycl_events(deps);
        if (send_buf != recv_buf) {
            LOG_DEBUG("single rank: out-of-place case, coll: alltoall");
            sycl_e = q.submit([=](sycl::handler& h) {
                h.depends_on(dep_events);
                h.memcpy(recv_buf, send_buf, count * ccl_dtype.size());
            });
        }
        else {
            LOG_DEBUG("single rank: inplace case, coll: alltoall");
            sycl_e = submit_wait_on_events(q, dep_events);
        }
        return ccl::event::create_from_native(sycl_e);
    }

    if (ccl::global_data::env().sycl_esimd) {
        LOG_WARN(
            "|CCL_SYCL| sycl ESIMD requested for alltoall collective; ESIMD not supported, falling back to alltoall sycl implementation");
    }

#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_begin("alltoall_large", "send_size", count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
    LOG_DEBUG("|CCL_SYCL| alltoall selects large kernel, count: ", count, " datatype: ", dtype);
    e = alltoall_large(send_buf, recv_buf, count, dtype, comm, global_stream, deps);
    LOG_DEBUG(
        "|CCL_SYCL| alltoall selects large kernel, count: ", count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT

    return e;
}

ccl::event alltoall_sycl(sycl::queue& q,
                         const void* send_buf,
                         void* recv_buf,
                         size_t count,
                         ccl::datatype dtype,
                         ccl_comm* comm,
                         ccl_stream* op_stream,
                         const alltoall_attr& attr,
                         const vector_class<event>& deps,
                         bool& done) {
    if (count == 0) {
        done = true;
        auto sycl_events = get_sycl_events(deps);
        auto e = submit_wait_on_events(q, sycl_events);
        return ccl::event::create_from_native(e);
    }

    bool is_single_node = false;
    if (ccl::global_data::env().backend == backend_mode::native) {
        const ccl::topo_manager& topo_manager = comm->get_topo_manager();
        is_single_node = topo_manager.is_single_node;
    }

    if (is_single_node) {

	if (is_arc_card(ccl::ze::get_device_family(op_stream->get_ze_device())) &&
                        ccl::global_data::env().sycl_enable_arc_alltoall_ll) {
            ccl::event e;
            uint32_t world = comm->get_node_comm()->size();
	    size_t dt_sz = ccl::global_data::get().dtypes->get(dtype).size();
            if (((count * dt_sz) % LS_SZ == 0) && ((world & (world-1)) == 0)) {
                done = true;
                e = arc_alltoall(send_buf, recv_buf, count, dtype, comm,op_stream);
            } else {
                done = false;
            }
	    return e;
        }

        if (send_buf != recv_buf) {
            LOG_DEBUG("is_single_node");
            return alltoall_sycl_single_node(
                q, send_buf, recv_buf, count, dtype, comm, op_stream, deps, done);
        }
        else {
            LOG_WARN(
                "|CCL_SYCL| sycl inplace requested for alltoall collective; inplace not supported, falling back");
        }
    }

    // multi-node scenario not supported, fallback
    done = false;
    return ccl::event();
}

} // namespace v1
} // namespace ccl
