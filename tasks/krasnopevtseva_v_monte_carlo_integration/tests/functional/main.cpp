#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>

#include "krasnopevtseva_v_monte_carlo_integration/common/include/common.hpp"
#include "krasnopevtseva_v_monte_carlo_integration/mpi/include/ops_mpi.hpp"
#include "krasnopevtseva_v_monte_carlo_integration/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace krasnopevtseva_v_monte_carlo_integration {

class KrasnopevtsevaVMCIntegrationFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &params = std::get<0>(test_param);
    double a = std::get<0>(params);
    double b = std::get<1>(params);
    int points = std::get<2>(params);
    std::uint8_t func = std::get<3>(params);

    std::string result = "a" + std::to_string(a) + "_b" + std::to_string(b) + "_points" + std::to_string(points) + "_" +
                         "_func" + std::to_string(func) + std::get<1>(test_param);

    std::ranges::replace(result, '-', 'n');
    std::ranges::replace(result, '.', '_');

    return result;
  }

 private:
  InType input_data_;
  double expected_integral_{};

 protected:
  void SetUp() override {
    auto test_param = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(test_param);

    double a = std::get<0>(input_data_);
    double b = std::get<1>(input_data_);
    std::uint8_t func = std::get<3>(input_data_);

    expected_integral_ = FuncSystem::AnalyticIntegral(func, a, b);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double a = std::get<0>(input_data_);
    double b = std::get<1>(input_data_);
    int points = std::get<2>(input_data_);
    std::uint8_t func = std::get<3>(input_data_);
    double tolerance = (b - a) / std::sqrt(points) * 10;
    if ((a <= -3.0) || (b >= 3.0)) {
      if (func <= 1) {
        tolerance *= 10;
      }
    }
    bool result = std::abs(output_data - expected_integral_) <= tolerance;
    if (!result) {
      std::cout << "diff=" << std::abs(output_data - expected_integral_) << ", tolerance=" << tolerance << '\n';
    }
    return result;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

namespace {

TEST_P(KrasnopevtsevaVMCIntegrationFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 20> kTestParam = {std::make_tuple(std::make_tuple(0.0, 1.0, 1000, 0), "cos_x3_small"),
                                             std::make_tuple(std::make_tuple(-2.0, 2.0, 10000, 0), "cos_x3_simmetr"),
                                             std::make_tuple(std::make_tuple(0.0, 3.0, 100000, 0), "cos_x3_large"),
                                             std::make_tuple(std::make_tuple(0.0, 0.0, 10, 0), "cos_x3_zero"),
                                             std::make_tuple(std::make_tuple(0.0, 1.0, 33333, 0), "cos_x3_odd_points"),

                                             std::make_tuple(std::make_tuple(0.0, 1.0, 1000, 1), "sin_x2_small"),
                                             std::make_tuple(std::make_tuple(-2.0, 2.0, 10000, 1), "sin_x2_simmetr"),
                                             std::make_tuple(std::make_tuple(0.0, 3.0, 100000, 1), "sin_x2_large"),
                                             std::make_tuple(std::make_tuple(0.0, 0.0, 10, 1), "sin_x2_zero"),
                                             std::make_tuple(std::make_tuple(0.0, 1.0, 99999, 1), "sin_x2_odd_points"),

                                             std::make_tuple(std::make_tuple(0.0, 1.0, 1000, 2), "exp_x_small"),
                                             std::make_tuple(std::make_tuple(-1.0, 1.0, 10000, 2), "exp_x_simmetr"),
                                             std::make_tuple(std::make_tuple(1.0, 3.0, 100000, 2), "exp_x_large"),
                                             std::make_tuple(std::make_tuple(0.0, 0.0, 10, 2), "exp_x_zero"),
                                             std::make_tuple(std::make_tuple(0.0, 1.0, 33333, 2), "exp_x_odd_points"),

                                             std::make_tuple(std::make_tuple(-1.0, 1.0, 1000, 3), "poly_symmetr"),
                                             std::make_tuple(std::make_tuple(0.0, 2.0, 10000, 3), "poly_positive"),
                                             std::make_tuple(std::make_tuple(-2.0, 2.0, 100000, 3), "poly_large"),
                                             std::make_tuple(std::make_tuple(0.0, 0.0, 10000, 3), "poly_zero"),
                                             std::make_tuple(std::make_tuple(0.0, 0.0, 99999, 3), "poly_odd_points")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KrasnopevtsevaVMCIntegrationMPI, InType>(
                                               kTestParam, PPC_SETTINGS_krasnopevtseva_v_monte_carlo_integration),
                                           ppc::util::AddFuncTask<KrasnopevtsevaVMCIntegrationSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_krasnopevtseva_v_monte_carlo_integration));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    KrasnopevtsevaVMCIntegrationFuncTests::PrintFuncTestName<KrasnopevtsevaVMCIntegrationFuncTests>;

INSTANTIATE_TEST_SUITE_P(MCIntegrationTests, KrasnopevtsevaVMCIntegrationFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace krasnopevtseva_v_monte_carlo_integration
