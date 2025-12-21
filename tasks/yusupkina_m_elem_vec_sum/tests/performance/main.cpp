#include <gtest/gtest.h>

#include <vector>

#include "util/include/perf_test_util.hpp"
#include "yusupkina_m_elem_vec_sum/common/include/common.hpp"
#include "yusupkina_m_elem_vec_sum/mpi/include/ops_mpi.hpp"
#include "yusupkina_m_elem_vec_sum/seq/include/ops_seq.hpp"

namespace yusupkina_m_elem_vec_sum {

class YusupkinaMRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100000000;
  InType input_data_;
  OutType expected_ = 0;

  void SetUp() override {
    static std::vector<int> static_data(kCount_);
    static bool first_time = true;

    if (first_time) {
      for (int i = 0; i < kCount_; i++) {
        static_data[i] = i;
      }
      first_time = false;
    }

    input_data_ = static_data;
    expected_ = static_cast<OutType>(kCount_) * (kCount_ - 1) / 2;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(YusupkinaMRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, YusupkinaMElemVecSumMPI, YusupkinaMElemVecSumSEQ>(
    PPC_SETTINGS_yusupkina_m_elem_vec_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = YusupkinaMRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, YusupkinaMRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace yusupkina_m_elem_vec_sum
