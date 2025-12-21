#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "zagryadskov_m_radix_sort_double_simple_merge/common/include/common.hpp"
#include "zagryadskov_m_radix_sort_double_simple_merge/mpi/include/radix_sort_double_simple_merge.hpp"
#include "zagryadskov_m_radix_sort_double_simple_merge/seq/include/radix_sort_double_simple_merge.hpp"

namespace zagryadskov_m_radix_sort_double_simple_merge {

class ZagryadskovMRunPerfTestRadixSortDoubleSimpleMerge : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    TestType param = 10'000'123;
    int seed = static_cast<int>(param % 100ULL);
    std::mt19937 e(seed);
    std::uniform_real_distribution<double> gen(-100000.0, 100000.0);
    std::vector<double> &vec = input_data_;
    vec.resize(param);
    std::ranges::generate(vec.begin(), vec.end(), [&]() { return gen(e); });
  }

  bool CheckTestOutputData(OutType &output_data) final {
    bool res = true;
    std::vector<double> example = input_data_;
    std::ranges::sort(example.begin(), example.end());
    if (example.size() != output_data.size()) {
      return false;
    }

    for (size_t j = 0; j < example.size(); ++j) {
      double diff = std::abs(example[j] - output_data[j]);
      double eps = std::max(std::abs(example[j]), std::abs(output_data[j])) * std::numeric_limits<double>::epsilon();
      if (diff > eps) {
        res = false;
      }
    }

    return res;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ZagryadskovMRunPerfTestRadixSortDoubleSimpleMerge, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ZagryadskovMRadixSortDoubleSimpleMergeMPI,
                                                       ZagryadskovMRadixSortDoubleSimpleMergeSEQ>(
    PPC_SETTINGS_zagryadskov_m_radix_sort_double_simple_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZagryadskovMRunPerfTestRadixSortDoubleSimpleMerge::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ZagryadskovMRunPerfTestRadixSortDoubleSimpleMerge, kGtestValues, kPerfTestName);

}  // namespace zagryadskov_m_radix_sort_double_simple_merge
