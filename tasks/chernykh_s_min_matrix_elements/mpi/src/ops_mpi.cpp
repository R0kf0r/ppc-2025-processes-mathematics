#include "chernykh_s_min_matrix_elements/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "chernykh_s_min_matrix_elements/common/include/common.hpp"

namespace chernykh_s_min_matrix_elements {

ChernykhSMinMatrixElementsMPI::ChernykhSMinMatrixElementsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = std::numeric_limits<double>::max();
}

bool ChernykhSMinMatrixElementsMPI::ValidationImpl() {
  return (GetOutput() == std::numeric_limits<double>::max());
}

bool ChernykhSMinMatrixElementsMPI::PreProcessingImpl() {
  return true;
}

bool ChernykhSMinMatrixElementsMPI::RunImpl() {
  int rank = 0;
  int size = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<double> full_data_buffer;
  std::vector<double> local_portion;
  size_t total_elements = 0;

  if (rank == 0) {
    const std::vector<std::vector<double>> &matrix = GetInput();

    for (const auto &row : matrix) {
      full_data_buffer.insert(full_data_buffer.end(), row.begin(), row.end());
    }
    total_elements = full_data_buffer.size();
  }

  int total_elements_int = static_cast<int>(total_elements);
  MPI_Bcast(&total_elements_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  total_elements = static_cast<size_t>(total_elements_int);

  if (total_elements == 0) {
    GetOutput() = std::numeric_limits<double>::max();
    return true;
  }

  int avg_elements_per_proc = static_cast<int>(total_elements / size);
  int remainder = static_cast<int>(total_elements % size);

  std::vector<int> elemcnt(size);
  std::vector<int> startpos(size);

  for (int i = 0; i < size; ++i) {
    elemcnt[i] = avg_elements_per_proc;
    if (i == size - 1) {
      elemcnt[i] += remainder;
    }
    if (i == 0) {
      startpos[i] = 0;
    } else {
      startpos[i] = startpos[i - 1] + elemcnt[i - 1];
    }
  }

  local_portion.resize(elemcnt[rank]);

  const double *send_buffer = nullptr;

  if (rank == 0) {
    send_buffer = full_data_buffer.data();
  }

  MPI_Scatterv(send_buffer, elemcnt.data(), startpos.data(), MPI_DOUBLE, local_portion.data(), elemcnt[rank],
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double local_min = std::numeric_limits<double>::max();

  for (double element : local_portion) {
    local_min = std::min(element, local_min);
  }

  double global_min = std::numeric_limits<double>::max();
  MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = global_min;

  return true;
}

bool ChernykhSMinMatrixElementsMPI::PostProcessingImpl() {
  return true;
}
}  // namespace chernykh_s_min_matrix_elements
