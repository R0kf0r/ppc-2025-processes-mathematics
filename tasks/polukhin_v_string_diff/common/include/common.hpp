#pragma once
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>

#include "task/include/task.hpp"

namespace polukhin_v_string_diff {

using InType = std::pair<std::string, std::string>;
using OutType = size_t;
using TestType = std::tuple<std::string, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace polukhin_v_string_diff
