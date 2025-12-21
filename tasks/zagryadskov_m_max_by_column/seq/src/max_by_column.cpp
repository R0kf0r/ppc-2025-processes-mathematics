#include "zagryadskov_m_max_by_column/seq/include/max_by_column.hpp"

#include <cstddef>
#include <limits>
#include <type_traits>

#include "zagryadskov_m_max_by_column/common/include/common.hpp"

namespace zagryadskov_m_max_by_column {

ZagryadskovMMaxByColumnSEQ::ZagryadskovMMaxByColumnSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ZagryadskovMMaxByColumnSEQ::ValidationImpl() {
  bool if_dividable = std::get<1>(GetInput()).size() % std::get<0>(GetInput()) == 0;
  return (std::get<0>(GetInput()) > 0) && (!std::get<1>(GetInput()).empty()) && (GetOutput().empty()) && if_dividable;
}

bool ZagryadskovMMaxByColumnSEQ::PreProcessingImpl() {
  bool if_dividable = std::get<1>(GetInput()).size() % std::get<0>(GetInput()) == 0;
  return (std::get<0>(GetInput()) > 0) && (!std::get<1>(GetInput()).empty()) && if_dividable;
}

bool ZagryadskovMMaxByColumnSEQ::RunImpl() {
  bool if_dividable = std::get<1>(GetInput()).size() % std::get<0>(GetInput()) == 0;
  if ((std::get<0>(GetInput()) == 0) || (std::get<1>(GetInput()).empty()) || !if_dividable) {
    return false;
  }

  const auto &n = std::get<0>(GetInput());
  const auto &mat = std::get<1>(GetInput());
  size_t m = mat.size() / n;
  OutType &res = GetOutput();
  OutType rows;
  using T = std::decay_t<decltype(*mat.begin())>;

  size_t j = 0;
  size_t i = 0;
  res.resize(n, std::numeric_limits<T>::lowest());
  T tmp = std::numeric_limits<T>::lowest();
  bool tmp_flag = false;
  for (j = 0; j < n; ++j) {
    for (i = 0; i < m; ++i) {
      tmp = mat[(j * m) + i];
      tmp_flag = tmp > res[j];
      res[j] = (static_cast<T>(tmp_flag) * tmp) + (static_cast<T>(!tmp_flag) * res[j]);
    }
  }

  return !GetOutput().empty();
}

bool ZagryadskovMMaxByColumnSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace zagryadskov_m_max_by_column
