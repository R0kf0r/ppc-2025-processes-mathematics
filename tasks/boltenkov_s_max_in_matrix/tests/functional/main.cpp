#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "boltenkov_s_max_in_matrix/common/include/common.hpp"
#include "boltenkov_s_max_in_matrix/mpi/include/ops_mpi.hpp"
#include "boltenkov_s_max_in_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace boltenkov_s_max_in_matrix {

class BoltenkovSRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string file_name = params + ".bin";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_boltenkov_s_max_in_matrix, file_name);
    std::ifstream file_stream(abs_path, std::ios::in | std::ios::binary);
    if (!file_stream.is_open()) {
      throw std::runtime_error("Error opening file!\n");
    }
    int m = -1;
    int n = -1;
    file_stream.read(reinterpret_cast<char *>(&m), sizeof(int));
    file_stream.read(reinterpret_cast<char *>(&n), sizeof(int));
    if (m <= 0 || n <= 0) {
      throw std::runtime_error("invalid input data!\n");
    }
    std::get<0>(input_data_) = n;
    std::vector<double> &v = std::get<1>(input_data_);
    v.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
    file_stream.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(sizeof(double) * m * n));
    file_stream.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::ranges::all_of(std::get<1>(input_data_), [&](auto elem) { return elem <= output_data; });
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(BoltenkovSRunFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {"matrix1", "matrix2", "matrix3"};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<BoltenkovSMaxInMatrixkMPI, InType>(kTestParam, PPC_SETTINGS_boltenkov_s_max_in_matrix),
    ppc::util::AddFuncTask<BoltenkovSMaxInMatrixkSEQ, InType>(kTestParam, PPC_SETTINGS_boltenkov_s_max_in_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BoltenkovSRunFuncTestsProcesses::PrintFuncTestName<BoltenkovSRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, BoltenkovSRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace boltenkov_s_max_in_matrix
