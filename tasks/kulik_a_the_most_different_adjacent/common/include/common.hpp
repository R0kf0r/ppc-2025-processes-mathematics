#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace kulik_a_the_most_different_adjacent {

using InType = std::vector<double>;
using OutType = std::pair<uint64_t, uint64_t>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kulik_a_the_most_different_adjacent
