#include "boltenkov_s_max_in_matrix/seq/include/ops_seq.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include "boltenkov_s_max_in_matrix/common/include/common.hpp"

namespace boltenkov_s_max_in_matrix {

BoltenkovSMaxInMatrixkSEQ::BoltenkovSMaxInMatrixkSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = -std::numeric_limits<double>::max();
}

bool BoltenkovSMaxInMatrixkSEQ::ValidationImpl() {
  return std::get<0>(GetInput()) > 0 && !std::get<1>(GetInput()).empty() &&
         std::get<1>(GetInput()).size() % std::get<0>(GetInput()) == 0;
}

bool BoltenkovSMaxInMatrixkSEQ::PreProcessingImpl() {
  return std::get<0>(GetInput()) > 0 && !std::get<1>(GetInput()).empty() &&
         std::get<1>(GetInput()).size() % std::get<0>(GetInput()) == 0;
}

bool BoltenkovSMaxInMatrixkSEQ::RunImpl() {
  if (std::get<0>(GetInput()) <= 0 || std::get<1>(GetInput()).empty() ||
      std::get<1>(GetInput()).size() % std::get<0>(GetInput()) != 0) {
    return false;
  }

  OutType &mx = GetOutput();
  std::vector<double> &v = std::get<1>(GetInput());

  int n = static_cast<int>(v.size());
  bool flag = false;

  for (int i = 0; i < n; ++i) {
    flag = v[i] > mx;
    mx = (static_cast<double>(flag) * v[i]) + (static_cast<double>(!flag) * mx);
  }

  return true;
}

bool BoltenkovSMaxInMatrixkSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace boltenkov_s_max_in_matrix
