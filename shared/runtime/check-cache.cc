#include <tvm/runtime/device_api.h>
#include <tvm/runtime/tensor.h>
#include "attn_utils.h"
#include <iostream>

using namespace tvm::runtime;
using namespace tvm::runtime::vm;

int main() {
  tvm::Device device{kDLCPU, 0};
  DLDataType dtype{kDLInt, 32, 1};
  PlainPagedKVCacheAuxDataManager cache(2, 128, 512, dtype, device, device, nullptr, true);
  HostMemoryVector full(2, dtype, device), sliding(2, dtype, device);
  full.push_back(64);
  sliding.push_back(32);
  int failures = 0;
  for (int depth = 0; depth < kPagedKVCacheMaxBlockDepth; ++depth) {
    auto check = [&](const char* name, const Tensor& global, const Tensor& local) {
      int first = static_cast<int32_t*>(global->data)[0];
      int second = static_cast<int32_t*>(local->data)[0];
      bool passed = first == 64 && second == 32;
      std::cout << name << " depth=" << depth << " full=" << first << " sliding=" << second
                << " " << (passed ? "PASS" : "FAIL") << "\n";
      failures += !passed;
    };
    int slot = depth + kPagedKVCacheMaxBlockDepth;
    auto full_pages = cache.CopyPageIndicesOnDepthAsync(&full, depth);
    auto local_pages = cache.CopyPageIndicesOnDepthAsync(&sliding, slot);
    check("pages", full_pages, local_pages);
    auto full_indptr = cache.CopyPageIndptrOnDepthAsync(&full, depth);
    auto local_indptr = cache.CopyPageIndptrOnDepthAsync(&sliding, slot);
    check("indptr", full_indptr, local_indptr);
    auto full_rope = cache.CopyKRoPEPosOffsetOnDepthAsync(&full, depth);
    auto local_rope = cache.CopyKRoPEPosOffsetOnDepthAsync(&sliding, slot);
    check("rope", full_rope, local_rope);
    auto full_length = cache.CopyLastPageLenOnDepthAsync(&full, depth);
    auto local_length = cache.CopyLengthInfoOnDepthAsync(&sliding, &sliding, &sliding, slot);
    check("length", full_length, local_length);
  }
  return failures != 0;
}
