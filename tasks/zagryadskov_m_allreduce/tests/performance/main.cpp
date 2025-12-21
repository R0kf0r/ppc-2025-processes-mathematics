#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "zagryadskov_m_allreduce/common/include/common.hpp"
#include "zagryadskov_m_allreduce/mpi/include/allreduce.hpp"
#include "zagryadskov_m_allreduce/seq/include/allreduce.hpp"

namespace zagryadskov_m_allreduce {

class ZagryadskovMRunPerfTestAllreduce : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    TestType params = 1;
    int seed = params;
    std::mt19937 e(seed);
    std::uniform_int_distribution<int> gen(-100, 100);
    int op = 0;
    int count = 25 + (params * 50'000'000);
    int processes = 4;

    std::get<1>(input_data_) = count;
    std::get<2>(input_data_) = op;
    auto &data_vec = std::get<0>(input_data_);
    data_vec.resize(static_cast<size_t>(count) * static_cast<size_t>(processes));

    for (int i = 0; i < count * processes; ++i) {
      data_vec[i] = gen(e);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    bool res = true;
    OutType example(output_data.size());

    int count = std::get<1>(input_data_);
    std::vector<int> in_data(count);
    MPI_Op op = ZagryadskovMAllreduceSEQ::GetOp(std::get<2>(input_data_));
    MPI_Scatter(std::get<0>(input_data_).data(), count, MPI_INT, in_data.data(), count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Allreduce(in_data.data(), example.data(), count, MPI_INT, op, MPI_COMM_WORLD);

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != example[i]) {
        res = false;
      }
    }

    return res;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ZagryadskovMRunPerfTestAllreduce, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ZagryadskovMAllreduceMPI, ZagryadskovMAllreduceSEQ>(
    PPC_SETTINGS_zagryadskov_m_allreduce);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZagryadskovMRunPerfTestAllreduce::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ZagryadskovMRunPerfTestAllreduce, kGtestValues, kPerfTestName);

}  // namespace zagryadskov_m_allreduce
