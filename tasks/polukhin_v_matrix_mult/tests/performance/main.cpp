#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "polukhin_v_matrix_mult/common/include/common.hpp"
#include "polukhin_v_matrix_mult/mpi/include/ops_mpi.hpp"
#include "polukhin_v_matrix_mult/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace polukhin_v_matrix_mult {

class PolukhinVRunPerfTestsMatrixMult : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const std::size_t dim = 700;

    std::seed_seq seed{42};
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    std::vector<double> mat_a(dim * dim);
    for (std::size_t i = 0; i < mat_a.size(); ++i) {
      mat_a[i] = dist(gen);
    }

    std::vector<double> mat_b(dim * dim);
    for (std::size_t i = 0; i < mat_b.size(); ++i) {
      mat_b[i] = dist(gen);
    }

    input_data_.matrix_a = mat_a;
    input_data_.matrix_b = mat_b;
    input_data_.dims.rows_a = dim;
    input_data_.dims.cols_a = dim;
    input_data_.dims.cols_b = dim;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &dims = input_data_.dims;
    const std::size_t expected_size = dims.rows_a * dims.cols_b;

    if (output_data.size() != expected_size) {
      return false;
    }

    bool has_non_zero = false;
    for (const auto &val : output_data) {
      if (std::isnan(val) || std::isinf(val)) {
        return false;
      }
      if (std::abs(val) > 1e-9) {
        has_non_zero = true;
      }
    }

    return has_non_zero;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(PolukhinVRunPerfTestsMatrixMult, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MatrixMultTaskMPI, MatrixMultTaskSEQ>(PPC_SETTINGS_polukhin_v_matrix_mult);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PolukhinVRunPerfTestsMatrixMult::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PolukhinVRunPerfTestsMatrixMult, kGtestValues, kPerfTestName);

}  // namespace polukhin_v_matrix_mult
