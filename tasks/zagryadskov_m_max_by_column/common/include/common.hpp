#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace zagryadskov_m_max_by_column {

using InType = std::tuple<size_t, std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace zagryadskov_m_max_by_column
