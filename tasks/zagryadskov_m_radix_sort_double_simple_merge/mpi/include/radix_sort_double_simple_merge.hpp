#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "zagryadskov_m_radix_sort_double_simple_merge/common/include/common.hpp"

namespace zagryadskov_m_radix_sort_double_simple_merge {

class ZagryadskovMRadixSortDoubleSimpleMergeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ZagryadskovMRadixSortDoubleSimpleMergeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void MyMPIMerge(std::vector<double> &data);
};

}  // namespace zagryadskov_m_radix_sort_double_simple_merge
