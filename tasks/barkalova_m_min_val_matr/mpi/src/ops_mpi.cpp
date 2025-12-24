#include "barkalova_m_min_val_matr/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "barkalova_m_min_val_matr/common/include/common.hpp"

namespace barkalova_m_min_val_matr {

BarkalovaMMinValMatrMPI::BarkalovaMMinValMatrMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput().resize(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    GetInput()[i] = in[i];
  }
  GetOutput().clear();
}

bool BarkalovaMMinValMatrMPI::ValidationImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  bool is_valid = true;

  if (rank == 0) {
    const auto &matrix = GetInput();
    if (!matrix.empty()) {
      size_t stolb = matrix[0].size();
      is_valid = std::ranges::all_of(matrix, [stolb](const auto &row) { return row.size() == stolb; });
    }
  }

  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return is_valid;
}

bool BarkalovaMMinValMatrMPI::PreProcessingImpl() {
  if (!GetInput().empty()) {
    size_t stolb = GetInput()[0].size();
    GetOutput().resize(stolb, INT_MAX);
  } else {
    GetOutput().clear();
  }
  return true;
}

namespace {

bool ValidateMatrixSize(int rank, size_t rows, size_t stolb) {
  bool size_valid = true;
  if (rank == 0) {
    if (rows > static_cast<size_t>(INT_MAX) || stolb > static_cast<size_t>(INT_MAX)) {
      size_valid = false;
    }

    if (size_valid && rows > 0 && stolb > 0) {
      if (rows > SIZE_MAX / stolb) {
        size_valid = false;
      } else {
        size_t total = rows * stolb;
        if (total > static_cast<size_t>(INT_MAX)) {
          size_valid = false;
        }
      }
    }
  }
  MPI_Bcast(&size_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return size_valid;
}

bool ValidateLocalSize(size_t rows, size_t col_stolb) {
  if (rows == 0 || col_stolb == 0) {
    return true;
  }

  if (rows > SIZE_MAX / col_stolb) {
    return false;
  }

  size_t total = rows * col_stolb;
  return total <= static_cast<size_t>(INT_MAX);
}

std::pair<size_t, size_t> GetMatrixDimensions(int rank, const std::vector<std::vector<int>> &matrix) {
  size_t rows = 0;
  size_t stolb = 0;

  if (rank == 0) {
    rows = matrix.size();
    stolb = matrix.empty() ? 0 : matrix[0].size();
  }

  std::array<uint64_t, 2> dims = {rows, stolb};
  MPI_Bcast(dims.data(), 2, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  return {dims[0], dims[1]};
}

std::pair<size_t, size_t> GetColumnRange(int rank, int size, size_t stolb) {
  if (stolb == 0) {
    return {0, 0};
  }
  size_t loc_stolb = stolb / static_cast<size_t>(size);
  size_t ostatok = stolb % static_cast<size_t>(size);

  size_t start_stolb = 0;
  for (int i = 0; i < rank; ++i) {
    size_t i_cols = loc_stolb + (std::cmp_less(i, ostatok) ? 1 : 0);
    start_stolb += i_cols;
  }
  size_t col_stolb = loc_stolb + (std::cmp_less(rank, ostatok) ? 1 : 0);
  return {start_stolb, col_stolb};
}

std::vector<int> PrepareDataForScatterv(int rank, const std::vector<std::vector<int>> &matrix, size_t rows,
                                        size_t stolb) {
  if (rank != 0) {
    return {};
  }
  std::vector<int> all_data(rows * stolb);
  for (size_t col = 0; col < stolb; ++col) {
    for (size_t row = 0; row < rows; ++row) {
      all_data[(col * rows) + row] = matrix[row][col];
    }
  }
  return all_data;
}

void PrepareScattervParams(int size, size_t rows, size_t stolb, std::vector<int> &send_counts,
                           std::vector<int> &displacements) {
  if (size == 0 || stolb == 0) {
    send_counts.clear();
    displacements.clear();
    return;
  }

  size_t base_cols = stolb / static_cast<size_t>(size);
  size_t extra_cols = stolb % static_cast<size_t>(size);

  send_counts.resize(size);
  displacements.resize(size);

  size_t current_displacement = 0;
  for (int i = 0; i < size; i++) {
    size_t i_cols = base_cols + (std::cmp_less(i, extra_cols) ? 1 : 0);
    send_counts[i] = static_cast<int>(i_cols * rows);
    displacements[i] = static_cast<int>(current_displacement);
    current_displacement += i_cols * rows;
  }
}

void DistributeDataScatterv(int rank, int size, const std::vector<int> &all_data, size_t rows, size_t stolb,
                            std::vector<int> &local_data, size_t &local_cols) {
  auto [start_col, col_count] = GetColumnRange(rank, size, stolb);
  local_cols = col_count;

  if (local_cols == 0 || rows == 0) {
    local_data.clear();
    return;
  }

  local_data.resize(rows * local_cols);

  std::vector<int> send_counts;
  std::vector<int> displacements;
  PrepareScattervParams(size, rows, stolb, send_counts, displacements);

  int recv_count = static_cast<int>(local_data.size());
  // распределяем матрицу по процессам
  MPI_Scatterv(all_data.empty() ? nullptr : all_data.data(), send_counts.data(), displacements.data(), MPI_INT,
               local_data.data(), recv_count, MPI_INT, 0, MPI_COMM_WORLD);
}

std::vector<int> CalculateLocalMins(const std::vector<int> &local_data, size_t rows, size_t local_cols) {
  if (local_cols == 0 || local_data.empty()) {
    return {};
  }

  std::vector<int> loc_min(local_cols, INT_MAX);

  for (size_t col = 0; col < local_cols; ++col) {
    size_t col_offset = col * rows;
    for (size_t row = 0; row < rows; ++row) {
      int value = local_data[col_offset + row];
      loc_min[col] = std::min(loc_min[col], value);
    }
  }

  return loc_min;
}

void PrepareGathervData(int size, size_t stolb, std::vector<int> &recv_counts, std::vector<int> &displacements) {
  size_t loc_stolb = stolb / static_cast<size_t>(size);
  size_t ostatok = stolb % static_cast<size_t>(size);

  recv_counts.resize(size);
  displacements.resize(size);

  size_t current_displacement = 0;
  for (int i = 0; i < size; i++) {
    size_t i_cols = loc_stolb + (std::cmp_less(i, ostatok) ? 1 : 0);
    recv_counts[i] = static_cast<int>(i_cols);
    displacements[i] = static_cast<int>(current_displacement);
    current_displacement += i_cols;
  }
}

void GatherAndBroadcastResults(const std::vector<int> &loc_min, int size, size_t stolb, size_t col_stolb,
                               std::vector<int> &res) {
  res.resize(stolb, INT_MAX);

  std::vector<int> recv_counts;
  std::vector<int> displacements;
  PrepareGathervData(size, stolb, recv_counts, displacements);

  int send_count = static_cast<int>(col_stolb);
  MPI_Gatherv(loc_min.data(), send_count, MPI_INT, res.data(), recv_counts.data(), displacements.data(), MPI_INT, 0,
              MPI_COMM_WORLD);

  MPI_Bcast(res.data(), static_cast<int>(stolb), MPI_INT, 0, MPI_COMM_WORLD);
}

}  // namespace

bool BarkalovaMMinValMatrMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &matrix = GetInput();
  auto [rows, stolb] = GetMatrixDimensions(rank, matrix);

  if (rows == 0 || stolb == 0) {
    GetOutput().clear();
    return true;
  }

  if (!ValidateMatrixSize(rank, rows, stolb)) {
    GetOutput().clear();
    return false;
  }

  auto [start_stolb, local_cols] = GetColumnRange(rank, size, stolb);

  if (!ValidateLocalSize(rows, local_cols)) {
    GetOutput().clear();
    return false;
  }
  std::vector<int> all_data;
  if (rank == 0) {
    all_data = PrepareDataForScatterv(rank, matrix, rows, stolb);
  }

  std::vector<int> local_data;
  DistributeDataScatterv(rank, size, all_data, rows, stolb, local_data, local_cols);

  std::vector<int> loc_min = CalculateLocalMins(local_data, rows, local_cols);

  GatherAndBroadcastResults(loc_min, size, stolb, local_cols, GetOutput());

  return true;
}

bool BarkalovaMMinValMatrMPI::PostProcessingImpl() {
  return true;
}
}  // namespace barkalova_m_min_val_matr
