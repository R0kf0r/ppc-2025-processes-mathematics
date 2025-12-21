#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace orehov_n_character_frequency {

using InType = std::tuple<std::string, std::string>;
using OutType = int;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace orehov_n_character_frequency
