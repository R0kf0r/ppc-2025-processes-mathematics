#pragma once

#include "petrov_e_find_max_in_columns_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace petrov_e_find_max_in_columns_matrix {

class PetrovEFindMaxInColumnsMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PetrovEFindMaxInColumnsMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace petrov_e_find_max_in_columns_matrix
