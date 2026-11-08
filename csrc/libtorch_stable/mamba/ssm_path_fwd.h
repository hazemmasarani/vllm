#pragma once

#ifndef USE_ROCM
  #include <cuda_bf16.h>
#else
  #include <hip/hip_bf16.h>
#endif
#include <cuda_fp16.h>

#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <type_traits>

struct SSMPathParams {
  using index_t = std::size_t;

  int batch;
  int dim;
  int seqlen;
  int dstate;
  int dim_ngroups_ratio;
  int64_t null_block_id;

  bool delta_softplus;
  bool cache_enabled;
  int block_size;

  index_t A_d_stride;
  index_t A_dstate_stride;
  index_t B_batch_stride;
  index_t B_d_stride;
  index_t B_dstate_stride;
  index_t B_group_stride;
  index_t C_batch_stride;
  index_t C_d_stride;
  index_t C_dstate_stride;
  index_t C_group_stride;
  index_t u_batch_stride;
  index_t u_d_stride;
  index_t delta_batch_stride;
  index_t delta_d_stride;
  index_t out_batch_stride;
  index_t out_d_stride;
  index_t ssm_states_batch_stride;
  index_t ssm_states_dim_stride;
  index_t ssm_states_dstate_stride;
  index_t cache_indices_stride;

  void* __restrict__ A_ptr;
  void* __restrict__ B_ptr;
  void* __restrict__ C_ptr;
  void* __restrict__ D_ptr;
  void* __restrict__ u_ptr;
  void* __restrict__ delta_ptr;
  void* __restrict__ delta_bias_ptr;
  void* __restrict__ out_ptr;
  void* __restrict__ ssm_states_ptr;

  void* __restrict__ query_start_loc_ptr;
  void* __restrict__ cache_indices_ptr;
  void* __restrict__ has_initial_state_ptr;

  void* __restrict__ block_idx_first_scheduled_token_ptr;
  void* __restrict__ block_idx_last_scheduled_token_ptr;
  void* __restrict__ initial_state_idx_ptr;
  void* __restrict__ cu_chunk_seqlen_ptr;
  void* __restrict__ last_chunk_indices_ptr;
};

namespace ssm_path_detail {

#ifndef USE_ROCM
constexpr std::size_t custom_max(std::initializer_list<std::size_t> values) {
  return std::max(values);
}

template <typename T>
constexpr T constexpr_min(T lhs, T rhs) {
  return std::min(lhs, rhs);
}
#else
constexpr std::size_t custom_max(std::initializer_list<std::size_t> values) {
  return *std::max_element(values.begin(), values.end());
}

template <typename T>
constexpr T constexpr_min(T lhs, T rhs) {
  return lhs < rhs ? lhs : rhs;
}
#endif

constexpr int kMaxDstate = 256;

template <int Bytes>
struct BytesToType {};

template <>
struct BytesToType<16> {
  using Type = uint4;
  static_assert(sizeof(Type) == 16);
};

template <>
struct BytesToType<8> {
  using Type = uint64_t;
  static_assert(sizeof(Type) == 8);
};

template <>
struct BytesToType<4> {
  using Type = uint32_t;
  static_assert(sizeof(Type) == 4);
};

template <>
struct BytesToType<2> {
  using Type = uint16_t;
  static_assert(sizeof(Type) == 2);
};

template <>
struct BytesToType<1> {
  using Type = uint8_t;
  static_assert(sizeof(Type) == 1);
};

template <typename scalar_t, int N>
struct Converter {
  static inline __device__ void to_float(const scalar_t (&src)[N],
                                         float (&dst)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = src[i];
    }
  }
};

template <int N>
struct Converter<torch::headeronly::Half, N> {
  static inline __device__ void to_float(
      const torch::headeronly::Half (&src)[N], float (&dst)[N]) {
    static_assert(N % 2 == 0);
    auto& src2 = reinterpret_cast<const half2(&)[N / 2]>(src);
    auto& dst2 = reinterpret_cast<float2(&)[N / 2]>(dst);
#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      dst2[i] = __half22float2(src2[i]);
    }
  }
};

#if __CUDA_ARCH__ >= 800
template <int N>
struct Converter<torch::headeronly::BFloat16, N> {
  static inline __device__ void to_float(
      const torch::headeronly::BFloat16 (&src)[N], float (&dst)[N]) {
    static_assert(N % 2 == 0);
    auto& src2 = reinterpret_cast<const nv_bfloat162(&)[N / 2]>(src);
    auto& dst2 = reinterpret_cast<float2(&)[N / 2]>(dst);
  #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      dst2[i] = __bfloat1622float2(src2[i]);
    }
  }
};
#endif

template <typename scalar_t>
struct ScanOp;

template <>
struct ScanOp<float> {
  __device__ __forceinline__ float2 operator()(const float2& ab0,
                                               const float2& ab1) const {
    return make_float2(ab1.x * ab0.x, ab1.x * ab0.y + ab1.y);
  }
};

template <typename scalar_t>
struct ScanPrefixCallbackOp {
  using scan_t =
      std::conditional_t<std::is_same_v<scalar_t, float>, float2, float4>;

  scan_t running_prefix;

  __device__ explicit ScanPrefixCallbackOp(scan_t running_prefix_)
      : running_prefix(running_prefix_) {}

  __device__ scan_t operator()(scan_t block_aggregate) {
    const scan_t old_prefix = running_prefix;
    running_prefix = ScanOp<scalar_t>()(running_prefix, block_aggregate);
    return old_prefix;
  }
};

template <typename Ktraits>
inline __device__ void load_input(
    typename Ktraits::input_t* input,
    typename Ktraits::input_t (&input_values)[Ktraits::kNItems],
    typename Ktraits::BlockLoadT::TempStorage& shared_load, int seqlen) {
  if constexpr (Ktraits::kIsEvenLen && !Ktraits::kVarlen) {
    auto& shared_load_vec =
        reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(
            shared_load);
    using vec_t = typename Ktraits::vec_t;
    typename Ktraits::BlockLoadVecT(shared_load_vec)
        .Load(reinterpret_cast<vec_t*>(input),
              reinterpret_cast<vec_t(&)[Ktraits::kNLoads]>(input_values)
#ifdef USE_ROCM
                  ,
              Ktraits::kNThreads * Ktraits::kNLoads
#endif
        );
  } else {
    typename Ktraits::BlockLoadT(shared_load)
        .Load(input, input_values, seqlen, 0.0f);
  }
}

template <typename Ktraits>
inline __device__ void load_weight(
    typename Ktraits::input_t* weight,
    typename Ktraits::weight_t (&weight_values)[Ktraits::kNItems],
    typename Ktraits::BlockLoadWeightT::TempStorage& shared_load, int seqlen) {
  constexpr int kNItems = Ktraits::kNItems;
  typename Ktraits::input_t loaded_values[kNItems];
  if constexpr (Ktraits::kIsEvenLen && !Ktraits::kVarlen) {
    auto& shared_load_vec =
        reinterpret_cast<typename Ktraits::BlockLoadWeightVecT::TempStorage&>(
            shared_load);
    using vec_t = typename Ktraits::vec_t;
    typename Ktraits::BlockLoadWeightVecT(shared_load_vec)
        .Load(reinterpret_cast<vec_t*>(weight),
              reinterpret_cast<vec_t(&)[Ktraits::kNLoads]>(loaded_values));
  } else {
    typename Ktraits::BlockLoadWeightT(shared_load)
        .Load(weight, loaded_values, seqlen, 0.0f);
  }
  Converter<typename Ktraits::input_t, kNItems>::to_float(loaded_values,
                                                          weight_values);
}

template <typename Ktraits>
inline __device__ void store_output(
    typename Ktraits::input_t* output,
    const float (&output_values)[Ktraits::kNItems],
    typename Ktraits::BlockStoreT::TempStorage& shared_store, int seqlen) {
  typename Ktraits::input_t write_values[Ktraits::kNItems];
#pragma unroll
  for (int i = 0; i < Ktraits::kNItems; ++i) {
    write_values[i] = output_values[i];
  }
  if constexpr (Ktraits::kIsEvenLen && !Ktraits::kVarlen) {
    auto& shared_store_vec =
        reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(
            shared_store);
    using vec_t = typename Ktraits::vec_t;
    typename Ktraits::BlockStoreVecT(shared_store_vec)
        .Store(reinterpret_cast<vec_t*>(output),
               reinterpret_cast<vec_t(&)[Ktraits::kNLoads]>(write_values));
  } else {
    typename Ktraits::BlockStoreT(shared_store)
        .Store(output, write_values, seqlen);
  }
}

}  // namespace ssm_path_detail

// Writes the ungated SSM result to delta.
void ssm_path_fwd(
    const torch::stable::Tensor& u, const torch::stable::Tensor& delta,
    const torch::stable::Tensor& A, const torch::stable::Tensor& B,
    const torch::stable::Tensor& C,
    const std::optional<torch::stable::Tensor>& D,
    const std::optional<torch::stable::Tensor>& delta_bias, bool delta_softplus,
    const std::optional<torch::stable::Tensor>& query_start_loc,
    const std::optional<torch::stable::Tensor>& cache_indices,
    const std::optional<torch::stable::Tensor>& has_initial_state,
    const torch::stable::Tensor& ssm_states, int64_t null_block_id,
    int64_t block_size,
    const std::optional<torch::stable::Tensor>& block_idx_first_scheduled_token,
    const std::optional<torch::stable::Tensor>& block_idx_last_scheduled_token,
    const std::optional<torch::stable::Tensor>& initial_state_idx,
    const std::optional<torch::stable::Tensor>& cu_chunk_seqlen,
    const std::optional<torch::stable::Tensor>& last_chunk_indices);
