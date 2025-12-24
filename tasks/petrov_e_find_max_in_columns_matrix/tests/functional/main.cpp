#include <gtest/gtest.h>

#include <array>
#include <cfloat>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "petrov_e_find_max_in_columns_matrix/common/include/common.hpp"
#include "petrov_e_find_max_in_columns_matrix/mpi/include/ops_mpi.hpp"
#include "petrov_e_find_max_in_columns_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace petrov_e_find_max_in_columns_matrix {

class PetrovERunFuncFindMaxInColumnsMatrix : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return "matrix_" + std::to_string(std::get<0>(test_param)) + "x" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    params_ = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::make_tuple(std::get<0>(params_), std::get<1>(params_), std::get<2>(params_));
    output_vector_ = std::get<3>(params_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == output_vector_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  TestType params_;
  InType input_data_;
  OutType output_vector_;
};

namespace {

TEST_P(PetrovERunFuncFindMaxInColumnsMatrix, FindMaxInColumns) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    TestType(3, 4, {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11}, {8, 9, 10, 11}),
    TestType(3, 3, {-1.251, -4.509, -1.902, -6.854, -1.950, -1.591, -2.147, -3.333, -10.054}, {-1.251, -1.591, -2.147}),
    TestType(2, 2, {0, 0, 0, 0}, {0, 0}),
    TestType(5, 1, {-24.49, -5.11, 29.45, 19.31, -84.5}, {29.45}),
    TestType(1, 5, {-24.49, -5.11, 29.45, 19.31, 84.5}, {-24.49, -5.11, 29.45, 19.31, 84.5}),
    TestType(0, 0, {}, {}),
    TestType(1, 1, {24}, {24}),
    TestType(5, 5, {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX,
                    DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX,
                    DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX},
             {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX}),
    TestType(4, 4, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, {1, 1, 1, 1}),
    TestType(4, 3, {0.0034, DBL_MAX, 0.0000045, 11, 0.0000000041, 0.000852, 0.1, 0.11, 1, 2, 3, 4},
             {DBL_MAX, 0.11, 4})};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<PetrovEFindMaxInColumnsMatrixMPI, InType>(
                                               kTestParam, PPC_SETTINGS_petrov_e_find_max_in_columns_matrix),
                                           ppc::util::AddFuncTask<PetrovEFindMaxInColumnsMatrixSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_petrov_e_find_max_in_columns_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    PetrovERunFuncFindMaxInColumnsMatrix::PrintFuncTestName<PetrovERunFuncFindMaxInColumnsMatrix>;

INSTANTIATE_TEST_SUITE_P(MaxInColumns, PetrovERunFuncFindMaxInColumnsMatrix, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace petrov_e_find_max_in_columns_matrix
