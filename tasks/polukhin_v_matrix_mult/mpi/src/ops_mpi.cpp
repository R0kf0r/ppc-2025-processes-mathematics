#include "polukhin_v_matrix_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "polukhin_v_matrix_mult/common/include/common.hpp"

namespace polukhin_v_matrix_mult {

namespace {

inline std::pair<int, int> ComputeRowsStart(int proc_index, int base_rows, int extra_rows) {
  if (proc_index < extra_rows) {
    const int rows_for_proc = base_rows + 1;
    const int start_for_proc = proc_index * (base_rows + 1);
    return {rows_for_proc, start_for_proc};
  }
  const int rows_for_proc = base_rows;
  const int start_for_proc = (extra_rows * (base_rows + 1)) + ((proc_index - extra_rows) * base_rows);
  return {rows_for_proc, start_for_proc};
}

inline void BroadcastMatrixDimensions(int rank, int &m, int &k, int &n) {
  std::array<int, 3> dims{m, k, n};
  MPI_Bcast(dims.data(), static_cast<int>(dims.size()), MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    m = dims[0];
    k = dims[1];
    n = dims[2];
  }
}

inline void DistributeMatrixB(int rank, std::vector<double> &b_matrix, int b_size) {
  if (rank != 0) {
    b_matrix.resize(static_cast<std::size_t>(b_size));
  }
  MPI_Bcast(b_matrix.data(), b_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

inline void PrepareScatterParameters(int size, int m, int k, std::vector<int> &sendcounts, std::vector<int> &displs) {
  sendcounts.assign(static_cast<std::size_t>(size), 0);
  displs.assign(static_cast<std::size_t>(size), 0);

  const int base_rows = (size == 0) ? 0 : (m / size);
  const int extra_rows = (size == 0) ? 0 : (m % size);

  for (int i = 0; i < size; ++i) {
    const auto pr = ComputeRowsStart(i, base_rows, extra_rows);
    const int rows_for_proc = pr.first;
    const int start_for_proc = pr.second;
    sendcounts[static_cast<std::size_t>(i)] = rows_for_proc * k;
    displs[static_cast<std::size_t>(i)] = start_for_proc * k;
  }
}

inline void PrepareGatherParameters(int size, int m, int n, std::vector<int> &recvcounts, std::vector<int> &rdispls) {
  recvcounts.assign(static_cast<std::size_t>(size), 0);
  rdispls.assign(static_cast<std::size_t>(size), 0);

  const int base_rows = (size == 0) ? 0 : (m / size);
  const int extra_rows = (size == 0) ? 0 : (m % size);

  for (int i = 0; i < size; ++i) {
    const auto pr = ComputeRowsStart(i, base_rows, extra_rows);
    const int rows_for_proc = pr.first;
    const int start_for_proc = pr.second;
    recvcounts[static_cast<std::size_t>(i)] = rows_for_proc * n;
    rdispls[static_cast<std::size_t>(i)] = start_for_proc * n;
  }
}

inline void ComputeLocalMatrixProduct(const std::vector<double> &local_a, const std::vector<double> &b_matrix,
                                      std::vector<double> &local_result, int my_rows, int k, int n) {
  const auto k_sz = static_cast<std::size_t>(k);
  const auto n_sz = static_cast<std::size_t>(n);

  for (int i = 0; i < my_rows; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int pk = 0; pk < k; ++pk) {
        const double val_a = local_a[(static_cast<std::size_t>(i) * k_sz) + static_cast<std::size_t>(pk)];
        const double val_b = b_matrix[(static_cast<std::size_t>(pk) * n_sz) + static_cast<std::size_t>(j)];
        sum += val_a * val_b;
      }
      local_result[(static_cast<std::size_t>(i) * n_sz) + static_cast<std::size_t>(j)] = sum;
    }
  }
}

}  // namespace

MatrixMultTaskMPI::MatrixMultTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool MatrixMultTaskMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &input = GetInput();
    const auto &dims = input.dims;

    if (dims.rows_a == 0 || dims.cols_a == 0 || dims.cols_b == 0) {
      return false;
    }

    const std::size_t expected_size_a = static_cast<std::size_t>(dims.rows_a) * static_cast<std::size_t>(dims.cols_a);
    const std::size_t expected_size_b = static_cast<std::size_t>(dims.cols_a) * static_cast<std::size_t>(dims.cols_b);

    if (input.matrix_a.size() != expected_size_a) {
      return false;
    }

    if (input.matrix_b.size() != expected_size_b) {
      return false;
    }
  }

  return true;
}

bool MatrixMultTaskMPI::PreProcessingImpl() {
  return true;
}

bool MatrixMultTaskMPI::PostProcessingImpl() {
  return true;
}

bool MatrixMultTaskMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int m = 0;
  int k = 0;
  int n = 0;

  std::vector<double> a_matrix;
  std::vector<double> b_matrix;

  if (rank == 0) {
    const auto &input = GetInput();
    a_matrix = input.matrix_a;
    b_matrix = input.matrix_b;
    m = static_cast<int>(input.dims.rows_a);
    k = static_cast<int>(input.dims.cols_a);
    n = static_cast<int>(input.dims.cols_b);
  }

  BroadcastMatrixDimensions(rank, m, k, n);

  const int b_size = k * n;
  DistributeMatrixB(rank, b_matrix, b_size);

  const int base_rows = (size == 0) ? 0 : (m / size);
  const int extra_rows = (size == 0) ? 0 : (m % size);

  const int my_rows = (rank < extra_rows) ? (base_rows + 1) : base_rows;

  std::vector<int> sendcounts;
  std::vector<int> displs;
  PrepareScatterParameters(size, m, k, sendcounts, displs);

  const int my_count = my_rows * k;
  std::vector<double> local_a(static_cast<std::size_t>(my_count));

  MPI_Scatterv(rank == 0 ? a_matrix.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE, local_a.data(),
               my_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_result(static_cast<std::size_t>(my_rows) * static_cast<std::size_t>(n));
  ComputeLocalMatrixProduct(local_a, b_matrix, local_result, my_rows, k, n);

  std::vector<int> recvcounts;
  std::vector<int> rdispls;
  PrepareGatherParameters(size, m, n, recvcounts, rdispls);

  if (rank == 0) {
    GetOutput().resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
  }

  const int my_result_count = my_rows * n;
  MPI_Gatherv(local_result.data(), my_result_count, MPI_DOUBLE, rank == 0 ? GetOutput().data() : nullptr,
              recvcounts.data(), rdispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (size > 1) {
    const int total_result_size = m * n;
    if (rank != 0) {
      GetOutput().resize(static_cast<std::size_t>(total_result_size));
    }
    MPI_Bcast(GetOutput().data(), total_result_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  return true;
}

}  // namespace polukhin_v_matrix_mult
