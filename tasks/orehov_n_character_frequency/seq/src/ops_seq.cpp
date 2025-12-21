#include "orehov_n_character_frequency/seq/include/ops_seq.hpp"

#include <cstddef>
#include <string>

#include "orehov_n_character_frequency/common/include/common.hpp"

namespace orehov_n_character_frequency {

OrehovNCharacterFrequencySEQ::OrehovNCharacterFrequencySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool OrehovNCharacterFrequencySEQ::ValidationImpl() {
  return (!std::get<0>(GetInput()).empty()) && (std::get<1>(GetInput()).length() == 1);
}

bool OrehovNCharacterFrequencySEQ::PreProcessingImpl() {
  return true;
}

bool OrehovNCharacterFrequencySEQ::RunImpl() {
  std::string str = std::get<0>(GetInput());
  std::string symbol = std::get<1>(GetInput());
  int result = 0;
  for (size_t i = 0; i < str.length(); i++) {
    if (str[i] == symbol[0]) {
      result++;
    }
  }
  GetOutput() = result;

  return true;
}

bool OrehovNCharacterFrequencySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace orehov_n_character_frequency
