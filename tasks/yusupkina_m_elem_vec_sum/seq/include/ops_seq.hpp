#pragma once

#include "task/include/task.hpp"
#include "yusupkina_m_elem_vec_sum/common/include/common.hpp"

namespace yusupkina_m_elem_vec_sum {

class YusupkinaMElemVecSumSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit YusupkinaMElemVecSumSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace yusupkina_m_elem_vec_sum
