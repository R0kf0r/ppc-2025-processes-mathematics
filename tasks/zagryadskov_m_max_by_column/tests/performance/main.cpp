#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <ios>
#include <iosfwd>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"
#include "zagryadskov_m_max_by_column/common/include/common.hpp"
#include "zagryadskov_m_max_by_column/mpi/include/max_by_column.hpp"
#include "zagryadskov_m_max_by_column/seq/include/max_by_column.hpp"

namespace zagryadskov_m_max_by_column {

class ZagryadskovMRunPerfTestMaxByColumn : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    std::string in_file_name = "mat1.bin";
    std::string abs_path =
        ppc::util::GetAbsoluteTaskPath(PPC_ID_zagryadskov_m_max_by_column,
                                       in_file_name);  // std::string abs_path = "../../data/mat1.bin";
    std::ifstream in_file_stream(abs_path, std::ios::in | std::ios::binary);
    if (!in_file_stream.is_open()) {
      throw std::runtime_error("Error opening file!\n");
    }
    size_t m = 0;
    size_t n = 0;
    in_file_stream.read(reinterpret_cast<char *>(&m), sizeof(size_t));
    in_file_stream.read(reinterpret_cast<char *>(&n), sizeof(size_t));
    std::get<0>(input_data_) = n;
    auto &mat = std::get<1>(input_data_);
    mat.resize(m * n);
    using T = std::decay_t<decltype(*mat.begin())>;

    in_file_stream.read(reinterpret_cast<char *>(mat.data()), static_cast<std::streamsize>(sizeof(T) * m * n));

    in_file_stream.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    bool res = true;
    size_t n = std::get<0>(input_data_);
    size_t m = std::get<1>(input_data_).size() / n;
    auto &mat = std::get<1>(input_data_);
    if (output_data.size() != n) {
      res = false;
      return res;
    }

    using T = std::decay_t<decltype(*mat.begin())>;
    OutType example(n, std::numeric_limits<T>::lowest());
    for (size_t j = 0; j < n; ++j) {
      for (size_t i = 0; i < m; ++i) {
        example[j] = std::max(example[j], mat[(j * m) + i]);
      }
    }

    for (size_t j = 0; j < n; ++j) {
      T diff = std::abs(example[j] - output_data[j]);
      T eps = std::max(std::abs(example[j]), std::abs(output_data[j])) * std::numeric_limits<double>::epsilon();
      if (diff > eps) {
        res = false;
      }
    }

    return res;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ZagryadskovMRunPerfTestMaxByColumn, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ZagryadskovMMaxByColumnMPI, ZagryadskovMMaxByColumnSEQ>(
    PPC_SETTINGS_zagryadskov_m_max_by_column);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZagryadskovMRunPerfTestMaxByColumn::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ZagryadskovMRunPerfTestMaxByColumn, kGtestValues, kPerfTestName);

}  // namespace zagryadskov_m_max_by_column
