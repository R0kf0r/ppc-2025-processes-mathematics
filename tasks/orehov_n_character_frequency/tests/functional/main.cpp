#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <string>
#include <tuple>

#include "orehov_n_character_frequency/common/include/common.hpp"
#include "orehov_n_character_frequency/mpi/include/ops_mpi.hpp"
#include "orehov_n_character_frequency/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace orehov_n_character_frequency {

class OrehovNCharacterFrequencyFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_orehov_n_character_frequency, params + ".txt");
    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw "file not open\n";
    }

    std::string str;
    std::string symbol;
    std::getline(file, str);
    std::getline(file, symbol);

    file.close();

    input_data_ = std::make_tuple(str, symbol);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int correct_result = 0;
    std::string str = std::get<0>(input_data_);
    std::string symbol = std::get<1>(input_data_);
    for (size_t i = 0; i < str.length(); i++) {
      if (str[i] == symbol[0]) {
        correct_result++;
      }
    }
    return correct_result == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(OrehovNCharacterFrequencyFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 1> kTestParam = {"string2"};  //, "string2", "string3"

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<OrehovNCharacterFrequencyMPI, InType>(kTestParam, PPC_SETTINGS_orehov_n_character_frequency),
    ppc::util::AddFuncTask<OrehovNCharacterFrequencySEQ, InType>(kTestParam,
                                                                 PPC_SETTINGS_orehov_n_character_frequency));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = OrehovNCharacterFrequencyFuncTests::PrintFuncTestName<OrehovNCharacterFrequencyFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, OrehovNCharacterFrequencyFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace orehov_n_character_frequency
