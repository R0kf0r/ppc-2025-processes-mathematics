#pragma once

#include "barkalova_m_min_val_matr/common/include/common.hpp"
#include "task/include/task.hpp"

namespace barkalova_m_min_val_matr {

class BarkalovaMMinValMatrMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BarkalovaMMinValMatrMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace barkalova_m_min_val_matr
