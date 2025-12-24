#pragma once

#include "chernykh_s_min_matrix_elements/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chernykh_s_min_matrix_elements {

class ChernykhSMinMatrixElementsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ChernykhSMinMatrixElementsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
}  // namespace chernykh_s_min_matrix_elements
