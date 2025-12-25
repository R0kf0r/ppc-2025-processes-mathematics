#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace polukhin_v_sparse_matmul_ccs {

struct SparseMatrixCCS {
  std::vector<double> values;
  std::vector<int> row_indices;
  std::vector<int> col_pointers;
  int rows;
  int cols;

  SparseMatrixCCS() : rows(0), cols(0) {}
  SparseMatrixCCS(int r, int c) : rows(r), cols(c) {
    col_pointers.resize(c + 1, 0);
  }
};

struct MatrixPair {
  SparseMatrixCCS matrix_a;
  SparseMatrixCCS matrix_b;
};

using InType = MatrixPair;
using OutType = SparseMatrixCCS;
using TestType = std::tuple<int, int, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace polukhin_v_sparse_matmul_ccs
