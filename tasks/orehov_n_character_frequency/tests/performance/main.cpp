#include <gtest/gtest.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <tuple>

#include "orehov_n_character_frequency/common/include/common.hpp"
#include "orehov_n_character_frequency/mpi/include/ops_mpi.hpp"
#include "orehov_n_character_frequency/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace orehov_n_character_frequency {

class OrehovNCharacterFrequencyPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_orehov_n_character_frequency, "string3.txt");
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
    int correct_result{};
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
};

TEST_P(OrehovNCharacterFrequencyPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, OrehovNCharacterFrequencyMPI, OrehovNCharacterFrequencySEQ>(
        PPC_SETTINGS_orehov_n_character_frequency);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = OrehovNCharacterFrequencyPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, OrehovNCharacterFrequencyPerfTests, kGtestValues, kPerfTestName);

}  // namespace orehov_n_character_frequency
