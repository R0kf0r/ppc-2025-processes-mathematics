#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "yusupkina_m_elem_vec_sum/common/include/common.hpp"
#include "yusupkina_m_elem_vec_sum/mpi/include/ops_mpi.hpp"
#include "yusupkina_m_elem_vec_sum/seq/include/ops_seq.hpp"

namespace yusupkina_m_elem_vec_sum {

class YusupkinaMElemVecSumFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int vec_size = std::get<0>(params);
    switch (vec_size) {
      case 0:
        input_data_ = {};
        expected_ = 0;
        break;
      case 1:
        input_data_ = {23};
        expected_ = 23;
        break;
      case 2:
        input_data_ = {10, 20};
        expected_ = 30;
        break;
      case 3:
        input_data_ = {0, 5, 0};
        expected_ = 5;
        break;
      case 4:
        input_data_ = {-1, 2, -3, 4};
        expected_ = 2;
        break;
      case 5:
        input_data_ = {100, -50, 25, -10, 5};
        expected_ = 70;
        break;
      case 6:
        input_data_ = std::vector<int>(6, 0);
        expected_ = 0;
        break;
      case 7:
        input_data_ = {1, 2, 3, 4, 5, 6, 7};
        expected_ = 28;
        break;
      case 8:
        input_data_ = std::vector<int>(8, 1);
        expected_ = 8;
        break;
      case 9:
        input_data_ = {1, -1, 1, -1, 1, -1, 1, -1, 1};
        expected_ = 1;
        break;
      case 10:
        input_data_ = std::vector<int>(10, -10);
        expected_ = -100;
        break;
      case 500:
        input_data_ = std::vector<int>(500, 2);
        expected_ = 1000;
        break;
      default:
        input_data_ = std::vector<int>(100000, 1);
        expected_ = 100000;
        break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_ = 0;
};

namespace {

TEST_P(YusupkinaMElemVecSumFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 13> kTestParam = {
    std::make_tuple(0, "empty_vector"),     std::make_tuple(1, "one_element"),      std::make_tuple(2, "two_elements"),
    std::make_tuple(3, "with_zeros"),       std::make_tuple(4, "negative_numbers"), std::make_tuple(5, "mixed_numbers"),
    std::make_tuple(6, "all_zeros"),        std::make_tuple(7, "sequential"),       std::make_tuple(8, "all_ones"),
    std::make_tuple(9, "alternating_sign"), std::make_tuple(10, "all_negative"),    std::make_tuple(500, "medium_size"),
    std::make_tuple(100000, "big_size")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<YusupkinaMElemVecSumMPI, InType>(kTestParam, PPC_SETTINGS_yusupkina_m_elem_vec_sum),
    ppc::util::AddFuncTask<YusupkinaMElemVecSumSEQ, InType>(kTestParam, PPC_SETTINGS_yusupkina_m_elem_vec_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = YusupkinaMElemVecSumFuncTests::PrintFuncTestName<YusupkinaMElemVecSumFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, YusupkinaMElemVecSumFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace yusupkina_m_elem_vec_sum
