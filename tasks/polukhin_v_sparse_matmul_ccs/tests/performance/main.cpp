#include <gtest/gtest.h>

#include <cstddef>
#include <map>
#include <random>
#include <vector>

#include "polukhin_v_sparse_matmul_ccs/common/include/common.hpp"
#include "polukhin_v_sparse_matmul_ccs/mpi/include/ops_mpi.hpp"
#include "polukhin_v_sparse_matmul_ccs/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace polukhin_v_sparse_matmul_ccs {

class PolukhinVRunPerfTestsSparseMatmulCCS : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const int size = 1000;
    const double density = 0.1;

    SparseMatrixCCS a = GenerateSparse(size, size, density);
    SparseMatrixCCS b = GenerateSparse(size, size, density);

    input_data_.matrix_a = a;
    input_data_.matrix_b = b;

    expected_output_ = ComputeExpected(a, b);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return CompareMatrices(expected_output_, output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;

  static SparseMatrixCCS GenerateSparse(int rows, int cols, double density) {
    SparseMatrixCCS mat(rows, cols);

    std::seed_seq seed{42};
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_real_distribution<double> val_dist(0.0, 10.0);

    for (int col = 0; col < cols; col++) {
      for (int row = 0; row < rows; row++) {
        double r = prob_dist(generator);
        if (r < density) {
          double val = val_dist(generator);
          mat.values.push_back(val);
          mat.row_indices.push_back(row);
        }
      }
      mat.col_pointers[col + 1] = static_cast<int>(mat.values.size());
    }

    return mat;
  }

  static SparseMatrixCCS ComputeExpected(const SparseMatrixCCS &a, const SparseMatrixCCS &b) {
    int res_rows = a.rows;
    int res_cols = b.cols;

    SparseMatrixCCS result(res_rows, res_cols);

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

    result.col_pointers[0] = 0;
    for (int col = 0; col < res_cols; col++) {
      for (const auto &item : temp_cols[col]) {
        if (std::abs(item.second) > 1e-10) {
          result.row_indices.push_back(item.first);
          result.values.push_back(item.second);
        }
      }
      result.col_pointers[col + 1] = static_cast<int>(result.values.size());
    }

    return result;
  }

  static bool CompareMatrices(const SparseMatrixCCS &expected, const SparseMatrixCCS &actual) {
    if (expected.rows != actual.rows || expected.cols != actual.cols) {
      return false;
    }

    if (expected.values.size() != actual.values.size()) {
      return false;
    }

    if (expected.row_indices.size() != actual.row_indices.size()) {
      return false;
    }

    if (expected.col_pointers.size() != actual.col_pointers.size()) {
      return false;
    }

    for (std::size_t i = 0; i < expected.values.size(); i++) {
      if (std::abs(expected.values[i] - actual.values[i]) > 1e-6) {
        return false;
      }
    }

    for (std::size_t i = 0; i < expected.row_indices.size(); i++) {
      if (expected.row_indices[i] != actual.row_indices[i]) {
        return false;
      }
    }

    for (std::size_t i = 0; i < expected.col_pointers.size(); i++) {
      if (expected.col_pointers[i] != actual.col_pointers[i]) {
        return false;
      }
    }

    return true;
  }
};

TEST_P(PolukhinVRunPerfTestsSparseMatmulCCS, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SparseMatmulCCSMPI, SparseMatmulCCSSEQ>(
    PPC_SETTINGS_polukhin_v_sparse_matmul_ccs);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PolukhinVRunPerfTestsSparseMatmulCCS::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PolukhinVRunPerfTestsSparseMatmulCCS, kGtestValues, kPerfTestName);

}  // namespace polukhin_v_sparse_matmul_ccs
