#pragma once

#include <torch/csrc/stable/tensor.h>

// Overwrites z with SiLU(z).
void gate_path_fwd(const torch::stable::Tensor& z);
