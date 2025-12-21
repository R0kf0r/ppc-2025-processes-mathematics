#include "zagryadskov_m_max_by_column/mpi/include/max_by_column.hpp"

#include <mpi.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "zagryadskov_m_max_by_column/common/include/common.hpp"

namespace zagryadskov_m_max_by_column {

ZagryadskovMMaxByColumnMPI::ZagryadskovMMaxByColumnMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int world_rank = 0;
  int err_code = 0;
  err_code = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_rank failed");
  }
  if (world_rank == 0) {
    GetInput() = in;
  }
}

bool ZagryadskovMMaxByColumnMPI::ValidationImpl() {
  bool res = false;
  int world_rank = 0;
  int err_code = 0;
  err_code = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_rank failed");
  }
  if (world_rank == 0) {
    size_t mat_size = std::get<1>(GetInput()).size();
    size_t n = std::get<0>(GetInput());
    bool if_dividable = mat_size % n == 0;
    bool if_suits_int = mat_size <= static_cast<size_t>(std::numeric_limits<int>::max());
    res = (n > 0) && (mat_size > 0) && (GetOutput().empty()) && if_dividable && if_suits_int;
  } else {
    res = true;
  }
  return res;
}

bool ZagryadskovMMaxByColumnMPI::PreProcessingImpl() {
  return true;
}

bool ZagryadskovMMaxByColumnMPI::SecondPhase(int m, int n, int world_size, int world_rank, std::vector<int> &sendcounts,
                                             std::vector<int> &displs, OutType &res, OutType &local_res,
                                             MPI_Datatype datatype) {
  int r = 0;
  int err_code = 0;
  for (r = 0; r < world_size; ++r) {
    sendcounts[r] /= m;
    if (r > 0) {
      displs[r] = displs[r - 1] + sendcounts[r - 1];
    }
  }

  err_code = MPI_Gatherv(local_res.data(), static_cast<int>(local_res.size()), datatype, res.data(), sendcounts.data(),
                         displs.data(), datatype, 0, MPI_COMM_WORLD);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Gatherv failed");
  }
  if (world_rank != 0) {
    res.resize(n);
  }
  // sequential version requires not to call MPI funcs
  err_code = MPI_Bcast(res.data(), static_cast<int>(res.size()), datatype, 0, MPI_COMM_WORLD);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Bcast failed");
  }

  bool result = false;
  if (world_rank == 0) {
    result = !res.empty();
  } else {
    result = true;
  }
  err_code = MPI_Barrier(MPI_COMM_WORLD);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Barrier failed");
  }
  return result;
}

bool ZagryadskovMMaxByColumnMPI::RunImpl() {
  int world_size = 0;
  int world_rank = 0;
  int err_code = 0;
  err_code = MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_size failed");
  }
  err_code = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_rank failed");
  }
  int n = 0;
  const void *mat_data = nullptr;
  int m = 0;
  OutType &res = GetOutput();
  OutType local_res;
  OutType columns;
  std::vector<int> sendcounts(world_size);
  std::vector<int> displs(world_size);
  if (!displs.empty()) {
    displs[0] = 0;
  }

  if (world_rank == 0) {
    n = static_cast<int>(std::get<0>(GetInput()));
    const auto &mat = std::get<1>(GetInput());
    m = static_cast<int>(mat.size()) / n;
    mat_data = reinterpret_cast<const void *>(mat.data());
  }
  err_code = MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Bcast failed");
  }
  err_code = MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Bcast failed");
  }

  int columns_count = n / world_size;
  using T = double;  // datatype cannot be sent to other processes
  MPI_Datatype datatype = MPI_DOUBLE;

  int i = 0;
  int j = 0;
  int r = 0;
  T tmp = std::numeric_limits<T>::lowest();
  bool tmp_flag = false;

  if (world_rank == 0) {
    res.assign(n, std::numeric_limits<T>::lowest());
  }
  for (r = 0; r < world_size; ++r) {
    sendcounts[r] = (columns_count + static_cast<int>(r < (n % world_size))) * m;
    if (r > 0) {
      displs[r] = displs[r - 1] + sendcounts[r - 1];
    }
  }

  local_res.assign(static_cast<size_t>(sendcounts[world_rank] / m), std::numeric_limits<T>::lowest());
  columns.resize(sendcounts[world_rank]);
  err_code = MPI_Scatterv(mat_data, sendcounts.data(), displs.data(), datatype, columns.data(), sendcounts[world_rank],
                          datatype, 0, MPI_COMM_WORLD);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Scatterv failed");
  }
  for (j = 0; std::cmp_less(j, local_res.size()); ++j) {
    for (i = 0; i < m; ++i) {
      tmp = columns[(j * m) + i];
      tmp_flag = tmp > local_res[j];
      local_res[j] = (static_cast<T>(tmp_flag) * tmp) + (static_cast<T>(!tmp_flag) * local_res[j]);
    }
  }

  return SecondPhase(m, n, world_size, world_rank, sendcounts, displs, res, local_res, datatype);
}

bool ZagryadskovMMaxByColumnMPI::PostProcessingImpl() {
  bool result = false;
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank == 0) {
    result = !GetOutput().empty();
  } else {
    result = true;
  }
  return result;
}

}  // namespace zagryadskov_m_max_by_column
