#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "barkalova_m_min_val_matr/common/include/common.hpp"
#include "barkalova_m_min_val_matr/mpi/include/ops_mpi.hpp"
#include "barkalova_m_min_val_matr/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace barkalova_m_min_val_matr {

class BarkalovaMMinValMatrPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    const size_t rows = 5000;
    const size_t stolb = 5000;

    std::vector<std::vector<int>> matrix(rows, std::vector<int>(stolb));

    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < stolb; ++j) {
        matrix[i][j] = static_cast<int>(i + j + 1);
      }
    }

    input_data_ = matrix;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &matrix = input_data_;

    if (output_data.empty()) {
      return false;
    }

    if (output_data.size() != matrix[0].size()) {
      return false;
    }

    std::vector<int> correct_result(matrix[0].size(), INT_MAX);
    for (const auto &row : matrix) {
      for (size_t j = 0; j < matrix[0].size(); ++j) {
        correct_result[j] = std::min(row[j], correct_result[j]);
      }
    }
    return output_data == correct_result;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(BarkalovaMMinValMatrPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, BarkalovaMMinValMatrMPI, BarkalovaMMinValMatrSEQ>(
    PPC_SETTINGS_barkalova_m_min_val_matr);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BarkalovaMMinValMatrPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BarkalovaMMinValMatrPerfTest, kGtestValues, kPerfTestName);

}  // namespace barkalova_m_min_val_matr
