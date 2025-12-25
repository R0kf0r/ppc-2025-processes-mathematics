#include "polukhin_v_matrix_mult/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "polukhin_v_matrix_mult/common/include/common.hpp"

namespace polukhin_v_matrix_mult {

MatrixMultTaskSEQ::MatrixMultTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}
bool MatrixMultTaskSEQ::ValidationImpl() {
  const auto &input = GetInput();
  const auto &dims = input.dims;

  if (dims.rows_a == 0 || dims.cols_a == 0 || dims.cols_b == 0) {
    return false;
  }

  const std::size_t expected_size_a = static_cast<std::size_t>(dims.rows_a) * static_cast<std::size_t>(dims.cols_a);
  const std::size_t expected_size_b = static_cast<std::size_t>(dims.cols_a) * static_cast<std::size_t>(dims.cols_b);
  if (input.matrix_a.size() != expected_size_a) {
    return false;
  }
  if (input.matrix_b.size() != expected_size_b) {
    return false;
  }
  return true;
}
bool MatrixMultTaskSEQ::PreProcessingImpl() {
  const auto &dims = GetInput().dims;
  const std::size_t result_size = static_cast<std::size_t>(dims.rows_a) * static_cast<std::size_t>(dims.cols_b);
  GetOutput().resize(result_size);

  return true;
}
bool MatrixMultTaskSEQ::RunImpl() {
  const auto &input = GetInput();
  const auto &a = input.matrix_a;
  const auto &b = input.matrix_b;
  const auto &dims = input.dims;
  auto &result = GetOutput();
  const std::size_t m = dims.rows_a;
  const std::size_t k = dims.cols_a;
  const std::size_t n = dims.cols_b;

  // алгоритм умножения матриц
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (std::size_t pk = 0; pk < k; ++pk) {
        const double a_elem = a[(i * k) + pk];
        const double b_elem = b[(pk * n) + j];
        sum += a_elem * b_elem;
      }
      result[(i * n) + j] = sum;
    }
  }
  return true;
}
bool MatrixMultTaskSEQ::PostProcessingImpl() {
  return true;
}
}  // namespace polukhin_v_matrix_mult
