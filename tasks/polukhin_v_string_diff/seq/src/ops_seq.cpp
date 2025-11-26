#include "polukhin_v_string_diff/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>

namespace polukhin_v_string_diff {

StringDiffTaskSEQ::StringDiffTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool StringDiffTaskSEQ::ValidationImpl() {
  return true;
}

bool StringDiffTaskSEQ::PreProcessingImpl() {
  return true;
}

bool StringDiffTaskSEQ::RunImpl() {
  const auto &input = GetInput();
  const std::string &str1 = input.first;
  const std::string &str2 = input.second;

  size_t result = 0;
  size_t min_len = std::min(str1.size(), str2.size());

  for (size_t i = 0; i < min_len; ++i) {
    if (str1[i] != str2[i]) {
      ++result;
    }
  }

  result += std::abs(static_cast<int>(str1.size()) - static_cast<int>(str2.size()));

  GetOutput() = result;
  return true;
}

bool StringDiffTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace polukhin_v_string_diff
