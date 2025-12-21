#include "lopatin_a_scalar_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "lopatin_a_scalar_mult/common/include/common.hpp"

namespace lopatin_a_scalar_mult {

LopatinAScalarMultMPI::LopatinAScalarMultMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  if (proc_rank == 0) {
    GetInput() = in;
  } else {
    GetInput() = InType{};
  }

  GetOutput() = 0.0;
}

bool LopatinAScalarMultMPI::ValidationImpl() {
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  if (proc_rank == 0) {
    return (!GetInput().first.empty() && !GetInput().second.empty()) &&
           (GetInput().first.size() == GetInput().second.size());
  }
  return true;
}

bool LopatinAScalarMultMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return (GetOutput() == 0.0);
}

bool LopatinAScalarMultMPI::RunImpl() {
  int proc_num = 0;
  int proc_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  const auto &input = GetInput();
  OutType &total_res = GetOutput();

  auto n = static_cast<uint64_t>(input.first.size());
  MPI_Bcast(&n, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  uint64_t local_n = n / static_cast<uint64_t>(proc_num);

  if (local_n > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("Too large input vector size!");
  }

  int local_n_int = static_cast<int>(local_n);

  if (local_n_int > 0) {
    InType local_data = std::make_pair(std::vector<double>(local_n_int), std::vector<double>(local_n_int));

    MPI_Scatter(input.first.data(), local_n_int, MPI_DOUBLE, local_data.first.data(), local_n_int, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    MPI_Scatter(input.second.data(), local_n_int, MPI_DOUBLE, local_data.second.data(), local_n_int, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    OutType proc_res{};
    for (int i = 0; i < local_n_int; ++i) {
      proc_res += local_data.first[i] * local_data.second[i];
    }

    MPI_Reduce(&proc_res, &total_res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }

  if (proc_rank == 0) {
    if (n % proc_num != 0) {
      uint64_t tail_index = n - (n % static_cast<uint64_t>(proc_num));
      for (uint64_t i = tail_index; i < n; ++i) {
        total_res += input.first[i] * input.second[i];
      }
    }
  }

  MPI_Bcast(&total_res, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

bool LopatinAScalarMultMPI::PostProcessingImpl() {
  return true;
}

}  // namespace lopatin_a_scalar_mult
