#include "yusupkina_m_elem_vec_sum/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "yusupkina_m_elem_vec_sum/common/include/common.hpp"

namespace yusupkina_m_elem_vec_sum {

YusupkinaMElemVecSumSEQ::YusupkinaMElemVecSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YusupkinaMElemVecSumSEQ::ValidationImpl() {
  return true;
}

bool YusupkinaMElemVecSumSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YusupkinaMElemVecSumSEQ::RunImpl() {
  GetOutput() = std::accumulate(GetInput().begin(), GetInput().end(), 0LL);
  return true;
}

bool YusupkinaMElemVecSumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace yusupkina_m_elem_vec_sum
