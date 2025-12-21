#pragma once

#include "polukhin_v_matrix_mult/common/include/common.hpp"
#include "task/include/task.hpp"

namespace polukhin_v_matrix_mult {

class MatrixMultTaskMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit MatrixMultTaskMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace polukhin_v_matrix_mult
