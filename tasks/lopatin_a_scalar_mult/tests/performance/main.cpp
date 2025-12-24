#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>

#include "lopatin_a_scalar_mult/common/include/common.hpp"
#include "lopatin_a_scalar_mult/mpi/include/ops_mpi.hpp"
#include "lopatin_a_scalar_mult/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace lopatin_a_scalar_mult {

class LopatinAScalarMultPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType output_chekup_data_{};

  void SetUp() override {
    std::string filename = "test_vectors_perf_n_4194304.bin";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_lopatin_a_scalar_mult, filename);
    std::ifstream infile(abs_path, std::ios::binary | std::ios::in);
    if (!infile.is_open()) {
      throw std::runtime_error("Failed to open file: " + filename);
    }

    int vector_size = 0;
    infile.read(reinterpret_cast<char *>(&vector_size), sizeof(vector_size));

    input_data_.first.resize(vector_size);
    input_data_.second.resize(vector_size);

    infile.read(reinterpret_cast<char *>(input_data_.first.data()),
                static_cast<std::streamsize>(vector_size * sizeof(double)));
    infile.read(reinterpret_cast<char *>(input_data_.second.data()),
                static_cast<std::streamsize>(vector_size * sizeof(double)));

    infile.read(reinterpret_cast<char *>(&output_chekup_data_), sizeof(output_chekup_data_));

    infile.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return abs(output_data - output_chekup_data_) < 0.1;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LopatinAScalarMultPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, LopatinAScalarMultMPI, LopatinAScalarMultSEQ>(
    PPC_SETTINGS_lopatin_a_scalar_mult);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LopatinAScalarMultPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(ScalarMultPerfTests, LopatinAScalarMultPerfTests, kGtestValues, kPerfTestName);

}  // namespace lopatin_a_scalar_mult
