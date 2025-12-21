#pragma once

#include <string>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace lopatin_a_scalar_mult {

using InType = std::pair<std::vector<double>, std::vector<double>>;
using OutType = double;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace lopatin_a_scalar_mult
