#pragma once

#include <cstdint>
#include <vector>

#include "kulik_a_the_most_different_adjacent/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kulik_a_the_most_different_adjacent {

class KulikATheMostDifferentAdjacentMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KulikATheMostDifferentAdjacentMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void CalculateDistribution(int proc_rank, int proc_num, uint64_t n, std::vector<int> &elemcnt,
                                    std::vector<int> &startpos);
  static void FindLocalMax(const std::vector<double> &buf, int start_index, double &max_diff_val,
                           uint64_t &max_diff_ind);
  static void CheckBoundaries(int proc_rank, int proc_num, const std::vector<int> &elemcnt,
                              const std::vector<int> &startpos, const std::vector<double> &buf, double &max_diff_val,
                              uint64_t &max_diff_ind);
};

}  // namespace kulik_a_the_most_different_adjacent
