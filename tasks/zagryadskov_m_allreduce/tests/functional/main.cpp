#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zagryadskov_m_allreduce/common/include/common.hpp"
#include "zagryadskov_m_allreduce/mpi/include/allreduce.hpp"
#include "zagryadskov_m_allreduce/seq/include/allreduce.hpp"

namespace zagryadskov_m_allreduce {

class ZagryadskovMRunFuncTestsAllreduce : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int seed = params;
    std::mt19937 e(seed);
    std::uniform_int_distribution<int> gen(-100, 100);
    int op = params;
    int count = 25 + (params * 1'000'000);
    int processes = 8;

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

 private:
  InType input_data_;
};

namespace {

TEST_P(ZagryadskovMRunFuncTestsAllreduce, GetAllreduce) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {0, 1, 2};  // 0 1

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ZagryadskovMAllreduceMPI, InType>(kTestParam, PPC_SETTINGS_zagryadskov_m_allreduce),
    ppc::util::AddFuncTask<ZagryadskovMAllreduceSEQ, InType>(kTestParam, PPC_SETTINGS_zagryadskov_m_allreduce));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ZagryadskovMRunFuncTestsAllreduce::PrintFuncTestName<ZagryadskovMRunFuncTestsAllreduce>;

INSTANTIATE_TEST_SUITE_P(AllreduceTests, ZagryadskovMRunFuncTestsAllreduce, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zagryadskov_m_allreduce
