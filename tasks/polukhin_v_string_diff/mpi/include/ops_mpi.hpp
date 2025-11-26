#pragma once

#include "polukhin_v_string_diff/common/include/common.hpp"
#include "task/include/task.hpp"

namespace polukhin_v_string_diff {

class StringDiffTaskMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit StringDiffTaskMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace polukhin_v_string_diff
