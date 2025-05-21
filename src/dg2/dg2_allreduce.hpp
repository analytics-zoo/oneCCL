#include "oneapi/ccl.hpp"
#include "coll/selection/selection.hpp"

#define DG2_NUM (4)

#define LL256_BUF_SIZE (32 * 1024 * 1024)
#define GATHER_BUF_OFFSET (LL256_BUF_SIZE / 2)

void dg2_init(ccl_selector_param &param,const void* send_buf,void* recv_buf);

ccl::event dg2_allreduce(const void *src, void *dst, size_t count,
                         const ccl_datatype& dtype, ccl::reduction reduction, ccl_comm *comm);

void dg2_clear();
