#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <tuple>

#include "kulik_a_the_most_different_adjacent/common/include/common.hpp"
#include "kulik_a_the_most_different_adjacent/mpi/include/ops_mpi.hpp"
#include "kulik_a_the_most_different_adjacent/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kulik_a_the_most_different_adjacent {

class KulikATheMostDifferentAdjacentFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string filename = params + ".bin";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_kulik_a_the_most_different_adjacent, filename);
    std::ifstream filestream(abs_path, std::ios::in | std::ios::binary);
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

 private:
  InType input_data_;
};

namespace {

TEST_P(KulikATheMostDifferentAdjacentFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::string("vector1"), std::string("vector2"), std::string("vector3")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KulikATheMostDifferentAdjacentMPI, InType>(
                                               kTestParam, PPC_SETTINGS_kulik_a_the_most_different_adjacent),
                                           ppc::util::AddFuncTask<KulikATheMostDifferentAdjacentSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_kulik_a_the_most_different_adjacent));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    KulikATheMostDifferentAdjacentFuncTests::PrintFuncTestName<KulikATheMostDifferentAdjacentFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, KulikATheMostDifferentAdjacentFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kulik_a_the_most_different_adjacent
