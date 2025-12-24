#pragma once

#include <cstddef>
#include <vector>

#include "task/include/task.hpp"

namespace zagryadskov_m_radix_sort_double_simple_merge {

using InType = std::vector<double>;
using OutType = std::vector<double>;
using TestType = size_t;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace zagryadskov_m_radix_sort_double_simple_merge
