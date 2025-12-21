#pragma once

#include <mpi.h>

#include "task/include/task.hpp"
#include "zagryadskov_m_allreduce/common/include/common.hpp"

namespace zagryadskov_m_allreduce {

class ZagryadskovMAllreduceSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ZagryadskovMAllreduceSEQ(const InType &in);

 private:
  OutType temp_vec_;
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 public:
  static MPI_Op GetOp(int iop);
};

}  // namespace zagryadskov_m_allreduce
