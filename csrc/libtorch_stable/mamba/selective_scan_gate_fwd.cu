#include "../torch_utils.h"

#include <torch/csrc/stable/macros.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kGateThreads = 256;
constexpr int kMaxGateBlocks = 4096;

template <typename input_t, int kRank>
__global__ void selective_scan_gate_fwd_kernel(input_t* z, int64_t numel,
                                               int64_t size_1, int64_t size_2,
                                               int64_t stride_0,
                                               int64_t stride_1,
                                               int64_t stride_2) {
  const int64_t thread_offset =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t grid_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (int64_t linear_idx = thread_offset; linear_idx < numel;
       linear_idx += grid_stride) {
    int64_t storage_offset;
    if constexpr (kRank == 2) {
      const int64_t row = linear_idx / size_1;
      const int64_t col = linear_idx - row * size_1;
      storage_offset = row * stride_0 + col * stride_1;
    } else {
      const int64_t plane_size = size_1 * size_2;
      const int64_t batch = linear_idx / plane_size;
      const int64_t plane_offset = linear_idx - batch * plane_size;
      const int64_t row = plane_offset / size_2;
      const int64_t col = plane_offset - row * size_2;
      storage_offset = batch * stride_0 + row * stride_1 + col * stride_2;
    }

    const float z_val = float(z[storage_offset]);
    z[storage_offset] = z_val / (1.0f + expf(-z_val));
  }
}

template <typename input_t>
void selective_scan_gate_fwd_cuda(const torch::stable::Tensor& z,
                                  cudaStream_t stream) {
  const int64_t numel = z.numel();
  if (numel == 0) {
    return;
  }

  const int64_t required_blocks = (numel + kGateThreads - 1) / kGateThreads;
  const int blocks = static_cast<int>(
      required_blocks < kMaxGateBlocks ? required_blocks : kMaxGateBlocks);
  auto* z_ptr = reinterpret_cast<input_t*>(z.data_ptr());

  if (z.dim() == 2) {
    selective_scan_gate_fwd_kernel<input_t, 2>
        <<<blocks, kGateThreads, 0, stream>>>(z_ptr, numel, z.size(1), 1,
                                              z.stride(0), z.stride(1), 0);
  } else {
    selective_scan_gate_fwd_kernel<input_t, 3>
        <<<blocks, kGateThreads, 0, stream>>>(z_ptr, numel, z.size(1),
                                              z.size(2), z.stride(0),
                                              z.stride(1), z.stride(2));
  }
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void selective_scan_gate_fwd(const torch::stable::Tensor& z) {
  const auto input_type = z.scalar_type();
  STD_TORCH_CHECK(
      input_type == torch::headeronly::ScalarType::Float ||
          input_type == torch::headeronly::ScalarType::Half ||
          input_type == torch::headeronly::ScalarType::BFloat16,
      "selective_scan_gate_fwd only supports float32, float16, and bfloat16");
  STD_TORCH_CHECK(z.is_cuda(), "z must be a CUDA tensor");
  STD_TORCH_CHECK(z.dim() == 2 || z.dim() == 3,
                  "z must be a 2D packed tensor or a 3D batched tensor");

  const torch::stable::accelerator::DeviceGuard device_guard(
      z.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  if (input_type == torch::headeronly::ScalarType::Half) {
    selective_scan_gate_fwd_cuda<torch::headeronly::Half>(z, stream);
  } else if (input_type == torch::headeronly::ScalarType::BFloat16) {
    selective_scan_gate_fwd_cuda<torch::headeronly::BFloat16>(z, stream);
  } else {
    selective_scan_gate_fwd_cuda<float>(z, stream);
  }
}
