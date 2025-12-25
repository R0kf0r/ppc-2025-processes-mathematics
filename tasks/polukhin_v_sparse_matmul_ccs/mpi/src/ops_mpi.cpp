#include "polukhin_v_sparse_matmul_ccs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "polukhin_v_sparse_matmul_ccs/common/include/common.hpp"

namespace polukhin_v_sparse_matmul_ccs {

SparseMatmulCCSMPI::SparseMatmulCCSMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = SparseMatrixCCS();
}

bool SparseMatmulCCSMPI::ValidationImpl() {
  return GetInput().matrix_a.cols == GetInput().matrix_b.rows;
}

bool SparseMatmulCCSMPI::PreProcessingImpl() {
  return true;
}

bool SparseMatmulCCSMPI::PostProcessingImpl() {
  return true;
}

namespace {

void BroadcastDimensions(int rank, SparseMatrixCCS &a, SparseMatrixCCS &b, int &res_rows, int &res_cols) {
  if (rank == 0) {
    res_rows = a.rows;
    res_cols = b.cols;
  }

  MPI_Bcast(&a.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&a.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&res_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&res_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void BroadcastMatrixA(int rank, SparseMatrixCCS &a) {
  int vals_size = 0;
  int rows_size = 0;
  int cols_size = 0;

  if (rank == 0) {
    vals_size = static_cast<int>(a.values.size());
    rows_size = static_cast<int>(a.row_indices.size());
    cols_size = static_cast<int>(a.col_pointers.size());
  }

  MPI_Bcast(&vals_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rows_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    a.values.resize(vals_size);
    a.row_indices.resize(rows_size);
    a.col_pointers.resize(cols_size);
  }

  if (vals_size > 0) {
    MPI_Bcast(a.values.data(), vals_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  if (rows_size > 0) {
    MPI_Bcast(a.row_indices.data(), rows_size, MPI_INT, 0, MPI_COMM_WORLD);
  }
  if (cols_size > 0) {
    MPI_Bcast(a.col_pointers.data(), cols_size, MPI_INT, 0, MPI_COMM_WORLD);
  }
}

// Вспомогательная функция для отправки столбцов процессу
void SendColumnRange(const SparseMatrixCCS &b, int dest, int dest_start, int dest_end) {
  if (dest_start >= b.cols) {
    dest_start = b.cols;
    dest_end = b.cols;
  }

  int start_idx = b.col_pointers[dest_start];
  int end_idx = (dest_end == b.cols) ? b.col_pointers[dest_end] : b.col_pointers[dest_end];
  int local_size = end_idx - start_idx;

  MPI_Send(&local_size, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
  if (local_size > 0) {
    MPI_Send(b.values.data() + start_idx, local_size, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
    MPI_Send(b.row_indices.data() + start_idx, local_size, MPI_INT, dest, 2, MPI_COMM_WORLD);
  }

  int col_ptr_size = dest_end - dest_start + 1;
  std::vector<int> adj_col_ptrs(col_ptr_size);
  for (int i = 0; i < col_ptr_size; i++) {
    adj_col_ptrs[i] = b.col_pointers[dest_start + i] - start_idx;
  }
  MPI_Send(adj_col_ptrs.data(), col_ptr_size, MPI_INT, dest, 3, MPI_COMM_WORLD);
}

void DistributeColumnsB_Root(int size, const SparseMatrixCCS &b, SparseMatrixCCS &local_b, int local_start,
                             int local_end, int cols_per_proc) {
  int start_idx = b.col_pointers[local_start];
  int end_idx = b.col_pointers[local_end];
  int local_size = end_idx - start_idx;

  local_b.values.assign(b.values.begin() + start_idx, b.values.begin() + end_idx);
  local_b.row_indices.assign(b.row_indices.begin() + start_idx, b.row_indices.begin() + end_idx);
  local_b.col_pointers.resize(local_b.cols + 1);
  for (int i = 0; i <= local_b.cols; i++) {
    local_b.col_pointers[i] = b.col_pointers[local_start + i] - start_idx;
  }

  for (int dest = 1; dest < size; dest++) {
    int dest_start = dest * cols_per_proc;
    int dest_end = std::min(dest_start + cols_per_proc, b.cols);
    SendColumnRange(b, dest, dest_start, dest_end);
  }
}

void DistributeColumnsB_NonRoot(int rank, SparseMatrixCCS &local_b) {
  int local_size = 0;
  MPI_Recv(&local_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (local_size > 0) {
    local_b.values.resize(local_size);
    local_b.row_indices.resize(local_size);
    MPI_Recv(local_b.values.data(), local_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(local_b.row_indices.data(), local_size, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  int col_ptr_size = local_b.cols + 1;
  local_b.col_pointers.resize(col_ptr_size);
  MPI_Recv(local_b.col_pointers.data(), col_ptr_size, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void DistributeColumnsB(int rank, int size, const SparseMatrixCCS &b, int &local_start, int &local_end,
                        SparseMatrixCCS &local_b) {
  int cols_per_proc = (b.cols + size - 1) / size;
  local_start = rank * cols_per_proc;
  local_end = std::min(local_start + cols_per_proc, b.cols);

  if (local_start >= b.cols) {
    local_start = b.cols;
    local_end = b.cols;
  }

  local_b.rows = b.rows;
  local_b.cols = local_end - local_start;

  if (rank == 0) {
    DistributeColumnsB_Root(size, b, local_b, local_start, local_end, cols_per_proc);
  } else {
    DistributeColumnsB_NonRoot(rank, local_b);
  }
}

void ComputeLocal(const SparseMatrixCCS &a, const SparseMatrixCCS &local_b, SparseMatrixCCS &local_res) {
  local_res.rows = a.rows;
  local_res.cols = local_b.cols;
  local_res.col_pointers.resize(local_res.cols + 1, 0);

  std::vector<std::map<int, double>> temp_cols(local_res.cols);

  for (int local_col = 0; local_col < local_b.cols; local_col++) {
    int start = local_b.col_pointers[local_col];
    int end = local_b.col_pointers[local_col + 1];

    for (int idx_b = start; idx_b < end; idx_b++) {
      int row_b = local_b.row_indices[idx_b];
      double val_b = local_b.values[idx_b];

      int col_a = row_b;
      if (col_a >= a.cols) {
        continue;
      }

      int start_a = a.col_pointers[col_a];
      int end_a = a.col_pointers[col_a + 1];

      for (int idx_a = start_a; idx_a < end_a; idx_a++) {
        int row_a = a.row_indices[idx_a];
        double val_a = a.values[idx_a];
        temp_cols[local_col][row_a] += val_a * val_b;
      }
    }
  }

  for (int col = 0; col < local_res.cols; col++) {
    for (const auto &item : temp_cols[col]) {
      if (std::abs(item.second) > 1e-10) {
        local_res.row_indices.push_back(item.first);
        local_res.values.push_back(item.second);
      }
    }
    local_res.col_pointers[col + 1] = static_cast<int>(local_res.values.size());
  }
}

void ReceiveProcessResult(int src, SparseMatrixCCS &received) {
  int recv_vals_size = 0;
  int recv_rows_size = 0;
  int recv_cols = 0;

  MPI_Recv(&recv_vals_size, 1, MPI_INT, src, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(&recv_rows_size, 1, MPI_INT, src, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(&recv_cols, 1, MPI_INT, src, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  received.cols = recv_cols;
  received.col_pointers.resize(recv_cols + 1);

  if (recv_vals_size > 0) {
    received.values.resize(recv_vals_size);
    received.row_indices.resize(recv_rows_size);
    MPI_Recv(received.values.data(), recv_vals_size, MPI_DOUBLE, src, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(received.row_indices.data(), recv_rows_size, MPI_INT, src, 8, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  MPI_Recv(received.col_pointers.data(), recv_cols + 1, MPI_INT, src, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void GatherResults_Root(int size, const SparseMatrixCCS &local_res, SparseMatrixCCS &final_res, int res_rows,
                        int res_cols) {
  final_res.rows = res_rows;
  final_res.cols = res_cols;
  final_res.col_pointers.resize(res_cols + 1, 0);

  std::vector<SparseMatrixCCS> all_locals;
  all_locals.reserve(size);
  all_locals.push_back(local_res);

  for (int src = 1; src < size; src++) {
    SparseMatrixCCS received;
    ReceiveProcessResult(src, received);
    all_locals.push_back(received);
  }

  int offset = 0;
  for (const auto &proc_res : all_locals) {
    for (int col = 0; col < proc_res.cols; col++) {
      int start = proc_res.col_pointers[col];
      int end = proc_res.col_pointers[col + 1];

      for (int idx = start; idx < end; idx++) {
        final_res.values.push_back(proc_res.values[idx]);
        final_res.row_indices.push_back(proc_res.row_indices[idx]);
      }
      final_res.col_pointers[offset + col + 1] = static_cast<int>(final_res.values.size());
    }
    offset += proc_res.cols;
  }
}

void GatherResults_NonRoot(int rank, const SparseMatrixCCS &local_res) {
  int send_vals_size = static_cast<int>(local_res.values.size());
  int send_rows_size = static_cast<int>(local_res.row_indices.size());
  int send_cols = local_res.cols;

  MPI_Send(&send_vals_size, 1, MPI_INT, 0, 4, MPI_COMM_WORLD);
  MPI_Send(&send_rows_size, 1, MPI_INT, 0, 5, MPI_COMM_WORLD);
  MPI_Send(&send_cols, 1, MPI_INT, 0, 6, MPI_COMM_WORLD);

  if (send_vals_size > 0) {
    MPI_Send(local_res.values.data(), send_vals_size, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD);
    MPI_Send(local_res.row_indices.data(), send_rows_size, MPI_INT, 0, 8, MPI_COMM_WORLD);
  }
  MPI_Send(local_res.col_pointers.data(), send_cols + 1, MPI_INT, 0, 9, MPI_COMM_WORLD);
}

void GatherResults(int rank, int size, const SparseMatrixCCS &local_res, SparseMatrixCCS &final_res, int res_rows,
                   int res_cols) {
  if (rank == 0) {
    GatherResults_Root(size, local_res, final_res, res_rows, res_cols);
  } else {
    GatherResults_NonRoot(rank, local_res);
  }
}

void BroadcastFinal(int rank, SparseMatrixCCS &final_res) {
  int vals_size = 0;
  int rows_size = 0;
  int cols_size = 0;

  if (rank == 0) {
    vals_size = static_cast<int>(final_res.values.size());
    rows_size = static_cast<int>(final_res.row_indices.size());
    cols_size = static_cast<int>(final_res.col_pointers.size());
  }

  MPI_Bcast(&final_res.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&final_res.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&vals_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rows_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    final_res.values.resize(vals_size);
    final_res.row_indices.resize(rows_size);
    final_res.col_pointers.resize(cols_size);
  }

  if (vals_size > 0) {
    MPI_Bcast(final_res.values.data(), vals_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  if (rows_size > 0) {
    MPI_Bcast(final_res.row_indices.data(), rows_size, MPI_INT, 0, MPI_COMM_WORLD);
  }
  if (cols_size > 0) {
    MPI_Bcast(final_res.col_pointers.data(), cols_size, MPI_INT, 0, MPI_COMM_WORLD);
  }
}

}  // namespace

bool SparseMatmulCCSMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto &input = GetInput();
  SparseMatrixCCS a = input.matrix_a;
  SparseMatrixCCS b = input.matrix_b;

  int res_rows = 0;
  int res_cols = 0;

  BroadcastDimensions(rank, a, b, res_rows, res_cols);
  BroadcastMatrixA(rank, a);

  int local_start = 0;
  int local_end = 0;
  SparseMatrixCCS local_b;

  DistributeColumnsB(rank, size, b, local_start, local_end, local_b);

  SparseMatrixCCS local_res;
  ComputeLocal(a, local_b, local_res);

  SparseMatrixCCS final_res;
  GatherResults(rank, size, local_res, final_res, res_rows, res_cols);

  BroadcastFinal(rank, final_res);

  GetOutput() = final_res;

  return true;
}

}  // namespace polukhin_v_sparse_matmul_ccs
