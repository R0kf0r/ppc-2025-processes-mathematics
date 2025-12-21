#include "kulik_a_the_most_different_adjacent/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kulik_a_the_most_different_adjacent/common/include/common.hpp"

namespace kulik_a_the_most_different_adjacent {

KulikATheMostDifferentAdjacentSEQ::KulikATheMostDifferentAdjacentSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KulikATheMostDifferentAdjacentSEQ::ValidationImpl() {
  return (GetInput().size() >= 2);
}

bool KulikATheMostDifferentAdjacentSEQ::PreProcessingImpl() {
  return true;
}

bool KulikATheMostDifferentAdjacentSEQ::RunImpl() {
  const auto &input = GetInput();
  const auto n = input.size();
  OutType &ans = GetOutput();
  double mx = 0.;
  uint64_t ind = 0;
  for (size_t i = 1; i < n; ++i) {
    if (std::abs(input[i - 1] - input[i]) > mx) {
      mx = std::abs(input[i - 1] - input[i]);
      ind = i - 1;
    }
  }
  ans.first = ind;
  ans.second = ind + 1;
  return true;
}

bool KulikATheMostDifferentAdjacentSEQ::PostProcessingImpl() {
  return (GetOutput().second == (GetOutput().first + 1));
}

}  // namespace kulik_a_the_most_different_adjacent
