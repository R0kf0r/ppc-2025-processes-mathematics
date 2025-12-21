#pragma once
#include <cstddef>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace polukhin_v_matrix_mult {

struct MatrixDimensions {
  size_t rows_a;
  size_t cols_a;
  size_t cols_b;
};

struct MatrixInput {
  std::vector<double> matrix_a;
  std::vector<double> matrix_b;
  MatrixDimensions dims;
};

using InType = MatrixInput;
using OutType = std::vector<double>;
using TestType = std::tuple<size_t, size_t, size_t>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace polukhin_v_matrix_mult
