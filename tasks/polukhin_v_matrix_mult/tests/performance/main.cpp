#include <gtest/gtest.h>

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
    // размер матриц для теста
    const size_t dim = 700;

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    // генерируем матрицу A
    std::vector<double> mat_a(dim * dim);
    for (size_t i = 0; i < mat_a.size(); ++i) {
      mat_a[i] = dist(gen);
    }

    // генерируем матрицу B
    std::vector<double> mat_b(dim * dim);
    for (size_t i = 0; i < mat_b.size(); ++i) {
      mat_b[i] = dist(gen);
    }

    input_data_.matrix_a = mat_a;
    input_data_.matrix_b = mat_b;
    input_data_.dims = {dim, dim, dim};

    // вычисляем эталонный результат
    expected_output_.resize(dim * dim);
    for (size_t i = 0; i < dim; ++i) {
      for (size_t j = 0; j < dim; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < dim; ++k) {
          sum += mat_a[i * dim + k] * mat_b[k * dim + j];
        }
        expected_output_[i * dim + j] = sum;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    double tolerance = 1e-3;
    for (size_t i = 0; i < output_data.size(); ++i) {
      double diff = std::abs(output_data[i] - expected_output_[i]);
      if (diff > tolerance) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
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
