#include "petrov_e_find_max_in_columns_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include "petrov_e_find_max_in_columns_matrix/common/include/common.hpp"

namespace petrov_e_find_max_in_columns_matrix {

PetrovEFindMaxInColumnsMatrixSEQ::PetrovEFindMaxInColumnsMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool PetrovEFindMaxInColumnsMatrixSEQ::ValidationImpl() {
  return (std::get<0>(GetInput()) * std::get<1>(GetInput()) == static_cast<int>(std::get<2>(GetInput()).size())) &&
         (GetOutput().empty());
}

bool PetrovEFindMaxInColumnsMatrixSEQ::PreProcessingImpl() {
  return (std::get<0>(GetInput()) * std::get<1>(GetInput()) == static_cast<int>(std::get<2>(GetInput()).size()));
}

bool PetrovEFindMaxInColumnsMatrixSEQ::RunImpl() {
  if ((std::get<0>(GetInput()) * std::get<1>(GetInput()) != static_cast<int>(std::get<2>(GetInput()).size()))) {
    return false;
  }

  auto &n = std::get<0>(GetInput());
  auto &m = std::get<1>(GetInput());
  auto &matrix = std::get<2>(GetInput());
  OutType &res = GetOutput();

  using MatrixElemType = std::remove_reference_t<decltype(matrix[0])>;

  int i = 0;
  int j = 0;

  res.resize(m);
  MatrixElemType max = NAN;

  for (i = 0; i < m; i++) {
    max = matrix[static_cast<int>(i * n)];
    for (j = 1; j < n; j++) {
      max = std::max(matrix[(i * n) + j], max);
    }
    res[i] = max;
  }

  return true;
}

bool PetrovEFindMaxInColumnsMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace petrov_e_find_max_in_columns_matrix
