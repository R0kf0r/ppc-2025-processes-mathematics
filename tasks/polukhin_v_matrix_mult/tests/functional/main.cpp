#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "polukhin_v_matrix_mult/common/include/common.hpp"
#include "polukhin_v_matrix_mult/mpi/include/ops_mpi.hpp"
#include "polukhin_v_matrix_mult/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace polukhin_v_matrix_mult {

class PolukhinVRunFuncTestsMatrixMult : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "x" + std::to_string(std::get<1>(test_param)) + "x" +
           std::to_string(std::get<2>(test_param));
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    size_t m = std::get<0>(params);
    size_t k = std::get<1>(params);
    size_t n = std::get<2>(params);

    // генерируем матрицу A
    std::vector<double> mat_a(m * k);
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < k; ++j) {
        mat_a[i * k + j] = static_cast<double>((i + 1) * (j + 1));
      }
    }

    // генерируем матрицу B
    std::vector<double> mat_b(k * n);
    for (size_t i = 0; i < k; ++i) {
      for (size_t j = 0; j < n; ++j) {
        double val = ((i + j + 1) % 3 == 0) ? 1.0 : 0.5;
        mat_b[i * n + j] = val;
      }
    }

    input_data_.matrix_a = mat_a;
    input_data_.matrix_b = mat_b;
    input_data_.dims = {m, k, n};

    // вычисляем ожидаемый результат
    expected_output_.resize(m * n);
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (size_t p = 0; p < k; ++p) {
          sum += mat_a[i * k + p] * mat_b[p * n + j];
        }
        expected_output_[i * n + j] = sum;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    double eps = 1e-6;
    for (size_t i = 0; i < output_data.size(); ++i) {
      double diff = std::abs(output_data[i] - expected_output_[i]);
      if (diff > eps) {
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

namespace {

TEST_P(PolukhinVRunFuncTestsMatrixMult, MatrixMultiplication) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 9> kTestParam = {
    std::make_tuple(1, 1, 1), std::make_tuple(2, 2, 2),    std::make_tuple(3, 3, 3),
    std::make_tuple(4, 4, 4), std::make_tuple(5, 3, 4),    std::make_tuple(10, 10, 10),
    std::make_tuple(7, 5, 6), std::make_tuple(15, 20, 10), std::make_tuple(20, 15, 25)};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<MatrixMultTaskMPI, InType>(kTestParam, PPC_SETTINGS_polukhin_v_matrix_mult),
                   ppc::util::AddFuncTask<MatrixMultTaskSEQ, InType>(kTestParam, PPC_SETTINGS_polukhin_v_matrix_mult));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = PolukhinVRunFuncTestsMatrixMult::PrintFuncTestName<PolukhinVRunFuncTestsMatrixMult>;

INSTANTIATE_TEST_SUITE_P(MatrixMultTests, PolukhinVRunFuncTestsMatrixMult, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace polukhin_v_matrix_mult
