#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace zagryadskov_m_allreduce {

using InType = std::tuple<std::vector<int>, int, int>;  // data, count of data for each processes, operation
using OutType = std::vector<int>;                       // vector on each process
using TestType = int;                                   // probably seed ot smth. Count of processes?
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace zagryadskov_m_allreduce
