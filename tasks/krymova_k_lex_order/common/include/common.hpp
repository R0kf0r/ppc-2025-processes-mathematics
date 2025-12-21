#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace krymova_k_lex_order {

using InType = std::tuple<std::string, std::string>;
using OutType = int;
using TestType = std::tuple<std::string, std::string, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace krymova_k_lex_order
