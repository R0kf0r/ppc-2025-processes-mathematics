#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>

#include "polukhin_v_string_diff/common/include/common.hpp"
#include "polukhin_v_string_diff/mpi/include/ops_mpi.hpp"
#include "polukhin_v_string_diff/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace polukhin_v_string_diff {

class PolukhinVRunFuncTestsStringDiff : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    const std::string &str1 = std::get<0>(params);
    const std::string &str2 = std::get<1>(params);

    input_data_ = std::make_pair(str1, str2);

    expected_output_ = 0;
    size_t min_len = std::min(str1.size(), str2.size());

    for (size_t i = 0; i < min_len; ++i) {
      if (str1[i] != str2[i]) {
        ++expected_output_;
      }
    }
    size_t len1 = str1.size();
    size_t len2 = str2.size();
    expected_output_ += (len1 > len2) ? (len1 - len2) : (len2 - len1);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_{};
};

namespace {

TEST_P(PolukhinVRunFuncTestsStringDiff, StringDiff) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 11> kTestParam = {std::make_tuple("abc", "abc"),
                                             std::make_tuple("abc", "abd"),
                                             std::make_tuple("abc", "abcd"),
                                             std::make_tuple("hello", "hallo"),
                                             std::make_tuple("abc", "acb"),
                                             std::make_tuple("", "a"),
                                             std::make_tuple("Helen", "hellen"),
                                             std::make_tuple("", ""),
                                             std::make_tuple("Alexander", "Alex"),
                                             std::make_tuple("BaRbArA", "bArBaRa"),
                                             std::make_tuple("MarcusAurelius", "MarkAureliy")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<StringDiffTaskMPI, InType>(kTestParam, PPC_SETTINGS_polukhin_v_string_diff),
                   ppc::util::AddFuncTask<StringDiffTaskSEQ, InType>(kTestParam, PPC_SETTINGS_polukhin_v_string_diff));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = PolukhinVRunFuncTestsStringDiff::PrintFuncTestName<PolukhinVRunFuncTestsStringDiff>;

INSTANTIATE_TEST_SUITE_P(StringDiffTests, PolukhinVRunFuncTestsStringDiff, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace polukhin_v_string_diff
