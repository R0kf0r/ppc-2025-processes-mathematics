#include "polukhin_v_matrix_mult/seq/include/ops_seq.hpp"

#include <algorithm>
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

  // проверяем что размеры не нулевые
  if (dims.rows_a == 0 || dims.cols_a == 0 || dims.cols_b == 0) {
    return false;
  }

  // проверяем соответствие размеров массивов
  size_t expected_size_a = dims.rows_a * dims.cols_a;
  size_t expected_size_b = dims.cols_a * dims.cols_b;

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
  size_t result_size = dims.rows_a * dims.cols_b;
  GetOutput().resize(result_size);
  // инициализация нулями
  std::fill(GetOutput().begin(), GetOutput().end(), 0.0);
  return true;
}

bool MatrixMultTaskSEQ::RunImpl() {
  const auto &input = GetInput();
  const auto &a = input.matrix_a;
  const auto &b = input.matrix_b;
  const auto &dims = input.dims;
  auto &result = GetOutput();

  size_t m = dims.rows_a;
  size_t k = dims.cols_a;
  size_t n = dims.cols_b;

  // стандартный алгоритм умножения матриц
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (size_t p = 0; p < k; ++p) {
        // элемент из i-й строки A и p-го столбца B
        double a_elem = a[i * k + p];
        double b_elem = b[p * n + j];
        sum += a_elem * b_elem;
      }
      result[i * n + j] = sum;
    }
  }

  return true;
}

bool MatrixMultTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace polukhin_v_matrix_mult
