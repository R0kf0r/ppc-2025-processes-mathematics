#pragma once

#include "krymova_k_lex_order/common/include/common.hpp"
#include "task/include/task.hpp"

namespace krymova_k_lex_order {

class KrymovaKLexSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KrymovaKLexSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace krymova_k_lex_order
