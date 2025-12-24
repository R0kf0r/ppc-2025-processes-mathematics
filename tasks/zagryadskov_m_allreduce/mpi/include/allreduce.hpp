#pragma once

#include <mpi.h>

#include "task/include/task.hpp"
#include "zagryadskov_m_allreduce/common/include/common.hpp"

namespace zagryadskov_m_allreduce {

class ZagryadskovMAllreduceMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ZagryadskovMAllreduceMPI(const InType &in);

 private:
  OutType temp_vec_;
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void ApplyOp(void *recvbuf, const void *tempbuf, int count, MPI_Datatype type, MPI_Op op, MPI_Comm comm);
  template <typename T>
  static void ApplyOp(void *recvbuf, const void *tempbuf, int count, MPI_Op op, MPI_Comm comm) {
    T *r = reinterpret_cast<T *>(recvbuf);
    const T *t = reinterpret_cast<const T *>(tempbuf);
    for (int i = 0; i < count; i++) {
      if (op == MPI_SUM) {
        r[i] += t[i];
      } else if (op == MPI_MAX) {
        r[i] = std::max(r[i], t[i]);
      } else if (op == MPI_MIN) {
        r[i] = std::min(r[i], t[i]);
      } else {
        MPI_Abort(comm, 1);
      }
    }
  }
  static int MyAllreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                         MPI_Comm comm);
};

}  // namespace zagryadskov_m_allreduce
