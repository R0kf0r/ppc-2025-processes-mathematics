#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "polukhin_v_sparse_matmul_ccs/common/include/common.hpp"
#include "polukhin_v_sparse_matmul_ccs/mpi/include/ops_mpi.hpp"
#include "polukhin_v_sparse_matmul_ccs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace polukhin_v_sparse_matmul_ccs {

class PolukhinVRunFuncTestsSparseMatmulCCS : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::to_string(std::get<1>(test_param)) + "_" +
           std::to_string(std::get<2>(test_param));
  }

 protected:
  void SetUp() override {
    std::seed_seq seed{42};
    std::mt19937 generator(seed);

    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int rows_a = std::get<0>(params);
    int cols_a = std::get<1>(params);
    int cols_b = std::get<2>(params);

    SparseMatrixCCS a = GenerateSparse(rows_a, cols_a, 0.3);
    SparseMatrixCCS b = GenerateSparse(cols_a, cols_b, 0.3);

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

    for (int col = 0; col < cols; col++) {
      for (int row = 0; row < rows; row++) {
        double r = static_cast<double>(rand()) / RAND_MAX;
        if (r < density) {
          double val = static_cast<double>(rand() % 100) / 10.0;
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

    for (size_t i = 0; i < expected.values.size(); i++) {
      if (std::abs(expected.values[i] - actual.values[i]) > 1e-6) {
        return false;
      }
    }

    for (size_t i = 0; i < expected.row_indices.size(); i++) {
      if (expected.row_indices[i] != actual.row_indices[i]) {
        return false;
      }
    }

    for (size_t i = 0; i < expected.col_pointers.size(); i++) {
      if (expected.col_pointers[i] != actual.col_pointers[i]) {
        return false;
      }
    }

    return true;
  }
};

namespace {

TEST_P(PolukhinVRunFuncTestsSparseMatmulCCS, SparseMatrixMultiplication) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {std::make_tuple(5, 5, 5), std::make_tuple(10, 10, 10),
                                            std::make_tuple(3, 4, 5), std::make_tuple(8, 6, 7),
                                            std::make_tuple(15, 10, 12)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SparseMatmulCCSMPI, InType>(kTestParam, PPC_SETTINGS_polukhin_v_sparse_matmul_ccs),
    ppc::util::AddFuncTask<SparseMatmulCCSSEQ, InType>(kTestParam, PPC_SETTINGS_polukhin_v_sparse_matmul_ccs));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    PolukhinVRunFuncTestsSparseMatmulCCS::PrintFuncTestName<PolukhinVRunFuncTestsSparseMatmulCCS>;

INSTANTIATE_TEST_SUITE_P(SparseMatmulTests, PolukhinVRunFuncTestsSparseMatmulCCS, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace polukhin_v_sparse_matmul_ccs
