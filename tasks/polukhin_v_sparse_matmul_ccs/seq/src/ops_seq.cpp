#include "polukhin_v_sparse_matmul_ccs/seq/include/ops_seq.hpp"

#include <cmath>
#include <map>
#include <vector>

#include "polukhin_v_sparse_matmul_ccs/common/include/common.hpp"

namespace polukhin_v_sparse_matmul_ccs {

SparseMatmulCCSSEQ::SparseMatmulCCSSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = SparseMatrixCCS();
}

bool SparseMatmulCCSSEQ::ValidationImpl() {
  const auto &input = GetInput();
  return input.matrix_a.cols == input.matrix_b.rows;
}

bool SparseMatmulCCSSEQ::PreProcessingImpl() {
  return true;
}

bool SparseMatmulCCSSEQ::PostProcessingImpl() {
  return true;
}

bool SparseMatmulCCSSEQ::RunImpl() {
  const auto &input = GetInput();
  const SparseMatrixCCS &a = input.matrix_a;
  const SparseMatrixCCS &b = input.matrix_b;

  int res_rows = a.rows;
  int res_cols = b.cols;

  GetOutput() = SparseMatrixCCS(res_rows, res_cols);
  auto &result = GetOutput();

  std::vector<std::map<int, double>> temp_cols(res_cols);

  for (int col_b = 0; col_b < b.cols; col_b++) {
    int start_b = b.col_pointers[col_b];
    int end_b = b.col_pointers[col_b + 1];

    for (int idx_b = start_b; idx_b < end_b; idx_b++) {
      int row_b = b.row_indices[idx_b];
      double val_b = b.values[idx_b];

      int col_a = row_b;
      int start_a = a.col_pointers[col_a];
      int end_a = a.col_pointers[col_a + 1];

      for (int idx_a = start_a; idx_a < end_a; idx_a++) {
        int row_a = a.row_indices[idx_a];
        double val_a = a.values[idx_a];
        temp_cols[col_b][row_a] += val_a * val_b;
      }
    }
  }

  for (int col = 0; col < res_cols; col++) {
    for (const auto &item : temp_cols[col]) {
      if (std::abs(item.second) > 1e-10) {
        result.row_indices.push_back(item.first);
        result.values.push_back(item.second);
      }
    }
    result.col_pointers[col + 1] = static_cast<int>(result.values.size());
  }

  return true;
}

}  // namespace polukhin_v_sparse_matmul_ccs
