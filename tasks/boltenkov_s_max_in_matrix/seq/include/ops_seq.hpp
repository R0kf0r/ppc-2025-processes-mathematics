#pragma once

#include "boltenkov_s_max_in_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace boltenkov_s_max_in_matrix {

class BoltenkovSMaxInMatrixkSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BoltenkovSMaxInMatrixkSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace boltenkov_s_max_in_matrix
