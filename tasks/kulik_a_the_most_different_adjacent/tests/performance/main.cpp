#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>

#include "kulik_a_the_most_different_adjacent/common/include/common.hpp"
#include "kulik_a_the_most_different_adjacent/mpi/include/ops_mpi.hpp"
#include "kulik_a_the_most_different_adjacent/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace kulik_a_the_most_different_adjacent {

class KulikATheMostDifferentAdjacentPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    std::string filename = "vector2.bin";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_kulik_a_the_most_different_adjacent, filename);
    std::ifstream filestream(abs_path, std::ios::binary | std::ios::in);
    if (!filestream.is_open()) {
      throw std::runtime_error("Failed to open file: " + filename);
    }
    size_t vector_size = 0;
    filestream.read(reinterpret_cast<char *>(&vector_size), sizeof(size_t));
    input_data_.resize(vector_size);
    filestream.read(reinterpret_cast<char *>(input_data_.data()),
                    static_cast<std::streamsize>(vector_size * sizeof(double)));
    filestream.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    size_t n = input_data_.size();
    bool check = true;
    double mx = std::abs(input_data_[output_data.first] - input_data_[output_data.second]);
    for (size_t i = 1; i < n; ++i) {
      if (std::abs(input_data_[i - 1] - input_data_[i]) - mx > 1e-12) {
        check = false;
      }
    }
    return check;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KulikATheMostDifferentAdjacentPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KulikATheMostDifferentAdjacentMPI, KulikATheMostDifferentAdjacentSEQ>(
        PPC_SETTINGS_kulik_a_the_most_different_adjacent);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KulikATheMostDifferentAdjacentPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KulikATheMostDifferentAdjacentPerfTests, kGtestValues, kPerfTestName);

}  // namespace kulik_a_the_most_different_adjacent
