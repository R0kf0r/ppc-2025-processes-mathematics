#include "chernykh_s_min_matrix_elements/seq/include/ops_seq.hpp"

#include <algorithm>
#include <limits>

#include "chernykh_s_min_matrix_elements/common/include/common.hpp"

namespace chernykh_s_min_matrix_elements {

ChernykhSMinMatrixElementsSEQ::ChernykhSMinMatrixElementsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = std::numeric_limits<double>::max();
}

bool ChernykhSMinMatrixElementsSEQ::ValidationImpl() {
  return (GetOutput() == std::numeric_limits<double>::max());
}

bool ChernykhSMinMatrixElementsSEQ::PreProcessingImpl() {
  return true;
}

bool ChernykhSMinMatrixElementsSEQ::RunImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    GetOutput() = std::numeric_limits<double>::max();
    return true;
  }
  double minimum = std::numeric_limits<double>::max();
  for (const auto &row : matrix) {
    for (double element : row) {
      minimum = std::min(element, minimum);
    }
  }

  GetOutput() = minimum;
  return true;
}

bool ChernykhSMinMatrixElementsSEQ::PostProcessingImpl() {
  return true;
  ;
}

}  // namespace chernykh_s_min_matrix_elements
