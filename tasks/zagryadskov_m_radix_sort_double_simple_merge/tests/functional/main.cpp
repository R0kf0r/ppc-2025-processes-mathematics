#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zagryadskov_m_radix_sort_double_simple_merge/common/include/common.hpp"
#include "zagryadskov_m_radix_sort_double_simple_merge/mpi/include/radix_sort_double_simple_merge.hpp"
#include "zagryadskov_m_radix_sort_double_simple_merge/seq/include/radix_sort_double_simple_merge.hpp"

namespace zagryadskov_m_radix_sort_double_simple_merge {

class ZagryadskovMRunFuncTestsRadixSortDoubleSimpleMerge
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(test_param);
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
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

 private:
  InType input_data_;
};

namespace {

TEST_P(ZagryadskovMRunFuncTestsRadixSortDoubleSimpleMerge, GetRadixSortDoubleSimpleMerge) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {10, 123, 1'000'000};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ZagryadskovMRadixSortDoubleSimpleMergeMPI, InType>(
                                               kTestParam, PPC_SETTINGS_zagryadskov_m_radix_sort_double_simple_merge),
                                           ppc::util::AddFuncTask<ZagryadskovMRadixSortDoubleSimpleMergeSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_zagryadskov_m_radix_sort_double_simple_merge));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ZagryadskovMRunFuncTestsRadixSortDoubleSimpleMerge::PrintFuncTestName<
    ZagryadskovMRunFuncTestsRadixSortDoubleSimpleMerge>;

INSTANTIATE_TEST_SUITE_P(RadixSortDoubleSimpleMergeTests, ZagryadskovMRunFuncTestsRadixSortDoubleSimpleMerge,
                         kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zagryadskov_m_radix_sort_double_simple_merge
