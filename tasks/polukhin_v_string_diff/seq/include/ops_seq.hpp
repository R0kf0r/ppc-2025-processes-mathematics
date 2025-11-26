#pragma once

#include "polukhin_v_string_diff/common/include/common.hpp"
#include "task/include/task.hpp"

namespace polukhin_v_string_diff {

class StringDiffTaskSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit StringDiffTaskSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace polukhin_v_string_diff
