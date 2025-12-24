#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <utility>

#include "polukhin_v_string_diff/common/include/common.hpp"
#include "polukhin_v_string_diff/mpi/include/ops_mpi.hpp"
#include "polukhin_v_string_diff/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace polukhin_v_string_diff {

class PolukhinVRunPerfTestsStringDiff : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const size_t size = 100000000;
    std::string long_str1(size, 'a');
    std::string long_str2(size, 'a');

    for (size_t i = 0; i < long_str2.size(); i += 10) {
      long_str2[i] = 'b';
    }

    input_data_ = std::make_pair(long_str1, long_str2);
    expected_output_ = 10000000;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_{};
};

TEST_P(PolukhinVRunPerfTestsStringDiff, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, StringDiffTaskMPI, StringDiffTaskSEQ>(PPC_SETTINGS_polukhin_v_string_diff);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PolukhinVRunPerfTestsStringDiff::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PolukhinVRunPerfTestsStringDiff, kGtestValues, kPerfTestName);

}  // namespace polukhin_v_string_diff
