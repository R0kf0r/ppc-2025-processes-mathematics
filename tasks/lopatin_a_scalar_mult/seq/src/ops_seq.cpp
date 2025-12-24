#include "lopatin_a_scalar_mult/seq/include/ops_seq.hpp"

#include <cstdint>
#include <vector>

#include "lopatin_a_scalar_mult/common/include/common.hpp"

namespace lopatin_a_scalar_mult {

LopatinAScalarMultSEQ::LopatinAScalarMultSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool LopatinAScalarMultSEQ::ValidationImpl() {
  return (!GetInput().first.empty() && !GetInput().second.empty()) &&
         (GetInput().first.size() == GetInput().second.size());
}

bool LopatinAScalarMultSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return (GetOutput() == 0.0);
}

bool LopatinAScalarMultSEQ::RunImpl() {
  const auto &input = GetInput();
  const auto n = static_cast<uint64_t>(input.first.size());
  OutType &res = GetOutput();

  for (uint64_t i = 0; i < n; ++i) {
    res += input.first[i] * input.second[i];
  }

  return true;
}

bool LopatinAScalarMultSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace lopatin_a_scalar_mult
