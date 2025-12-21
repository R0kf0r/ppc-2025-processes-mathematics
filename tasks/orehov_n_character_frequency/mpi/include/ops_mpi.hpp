#pragma once

#include "orehov_n_character_frequency/common/include/common.hpp"
#include "task/include/task.hpp"

namespace orehov_n_character_frequency {

class OrehovNCharacterFrequencyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit OrehovNCharacterFrequencyMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace orehov_n_character_frequency
