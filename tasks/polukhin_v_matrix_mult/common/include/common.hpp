#pragma once
#include <cstddef>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace polukhin_v_matrix_mult {

struct MatrixDimensions {
  std::size_t rows_a = 0;
  std::size_t cols_a = 0;
  std::size_t cols_b = 0;
};

struct MatrixInput {
  std::vector<double> matrix_a;
  std::vector<double> matrix_b;
  MatrixDimensions dims{};
};

using InType = MatrixInput;
using OutType = std::vector<double>;
using TestType = std::tuple<std::size_t, std::size_t, std::size_t>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace polukhin_v_matrix_mult
