#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <tuple>

#include "krasnopevtseva_v_monte_carlo_integration/common/include/common.hpp"
#include "krasnopevtseva_v_monte_carlo_integration/mpi/include/ops_mpi.hpp"
#include "krasnopevtseva_v_monte_carlo_integration/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace krasnopevtseva_v_monte_carlo_integration {

class KrasnopevtsevaVMCIntegrationPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  double expected_integral_{};
  double tolerance_{};

  void SetUp() override {
    double a = 0.0;
    double b = 2.0;
    int points = 50000000;
    std::uint8_t func = 0;
    input_data_ = std::make_tuple(a, b, points, func);
    tolerance_ = (b - a) / std::sqrt(points) * 10;
    expected_integral_ = FuncSystem::AnalyticIntegral(func, a, b);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::abs(output_data - expected_integral_) < tolerance_;
  }
  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KrasnopevtsevaVMCIntegrationPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KrasnopevtsevaVMCIntegrationMPI, KrasnopevtsevaVMCIntegrationSEQ>(
        PPC_SETTINGS_krasnopevtseva_v_monte_carlo_integration);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KrasnopevtsevaVMCIntegrationPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KrasnopevtsevaVMCIntegrationPerfTests, kGtestValues, kPerfTestName);

}  // namespace krasnopevtseva_v_monte_carlo_integration
