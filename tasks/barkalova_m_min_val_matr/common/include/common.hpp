#pragma once

// #include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace barkalova_m_min_val_matr {

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<std::vector<std::vector<int>>, std::vector<int>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace barkalova_m_min_val_matr
