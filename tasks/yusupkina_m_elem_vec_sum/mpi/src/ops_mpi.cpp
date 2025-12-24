#include "yusupkina_m_elem_vec_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "yusupkina_m_elem_vec_sum/common/include/common.hpp"

namespace yusupkina_m_elem_vec_sum {

YusupkinaMElemVecSumMPI::YusupkinaMElemVecSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YusupkinaMElemVecSumMPI::ValidationImpl() {
  return true;
}

bool YusupkinaMElemVecSumMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YusupkinaMElemVecSumMPI::RunImpl() {
  int rank = 0;
  int count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &count);

  int vec_size = 0;

  if (rank == 0) {
    vec_size = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&vec_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (vec_size == 0) {
    GetOutput() = 0;
    return true;
  }

  int base_size = vec_size / count;
  int remainder = vec_size % count;
  int cur_size = base_size + (rank < remainder ? 1 : 0);

  std::vector<int> sendcounts(count);
  std::vector<int> displs(count);
  int start_ind = 0;
  for (int i = 0; i < count; ++i) {
    sendcounts[i] = base_size + (i < remainder ? 1 : 0);
    displs[i] = start_ind;
    start_ind += sendcounts[i];
  }

  int *sendbuf = nullptr;
  if (rank == 0) {
    sendbuf = GetInput().data();
  }
  std::vector<int> local_data(cur_size);
  MPI_Scatterv(sendbuf, sendcounts.data(), displs.data(), MPI_INT, local_data.data(), cur_size, MPI_INT, 0,
               MPI_COMM_WORLD);

  OutType local_sum = 0LL;
  for (int i = 0; i < cur_size; ++i) {
    local_sum += static_cast<OutType>(local_data[i]);
  }
  OutType global_sum = 0LL;

  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = global_sum;
  return true;
}

bool YusupkinaMElemVecSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace yusupkina_m_elem_vec_sum
