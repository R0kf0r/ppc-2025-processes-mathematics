#pragma once

#include "krasnopevtseva_v_monte_carlo_integration/common/include/common.hpp"
#include "task/include/task.hpp"

namespace krasnopevtseva_v_monte_carlo_integration {

class KrasnopevtsevaVMCIntegrationMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KrasnopevtsevaVMCIntegrationMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace krasnopevtseva_v_monte_carlo_integration
