#pragma once

#include "polukhin_v_sparse_matmul_ccs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace polukhin_v_sparse_matmul_ccs {

class SparseMatmulCCSSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SparseMatmulCCSSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace polukhin_v_sparse_matmul_ccs
