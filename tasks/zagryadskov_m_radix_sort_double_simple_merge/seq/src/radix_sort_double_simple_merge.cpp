#include "zagryadskov_m_radix_sort_double_simple_merge/seq/include/radix_sort_double_simple_merge.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "zagryadskov_m_radix_sort_double_simple_merge/common/include/common.hpp"

namespace zagryadskov_m_radix_sort_double_simple_merge {

ZagryadskovMRadixSortDoubleSimpleMergeSEQ::ZagryadskovMRadixSortDoubleSimpleMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ZagryadskovMRadixSortDoubleSimpleMergeSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool ZagryadskovMRadixSortDoubleSimpleMergeSEQ::PreProcessingImpl() {
  return true;
}

void ZagryadskovMRadixSortDoubleSimpleMergeSEQ::Foffset(const uint8_t *mas, size_t size, size_t offset,
                                                        std::array<uint64_t, 256> &count) {
  size_t i = 0;
  uint64_t tmp = 0;
  memset(count.data(), 0, (255ULL + 1ULL) * sizeof(uint64_t));
  for (i = 0; i < size * sizeof(double); i += sizeof(double)) {
    count.at(mas[i + offset])++;
  }
  tmp = count[0ULL];
  count[0ULL] = 0ULL;
  for (i = 0ULL; i < 255ULL; i++) {
    std::swap(tmp, count.at(i + 1));
    count.at(i + 1) += count.at(i);
  }
}

void ZagryadskovMRadixSortDoubleSimpleMergeSEQ::RadixSortLSD(double *mas, size_t size) {
  size_t i = 0;
  size_t j = 0;
  size_t k = 0;
  size_t tidy_const = 0;
  uint8_t *pm = nullptr;
  std::array<uint64_t, 256> count{};
  std::vector<double> vec_buf(size);
  double *mas2 = vec_buf.data();
  pm = reinterpret_cast<uint8_t *>(mas);

  for (i = 0ULL; i < sizeof(double); i++) {
    Foffset(pm, size, i, count);
    for (j = 0ULL; j < size; j++) {
      tidy_const = count.at(pm[(j * sizeof(double)) + i]);
      mas2[tidy_const] = mas[j];
      count.at(pm[(j * sizeof(double)) + i])++;
    }
    std::swap(mas, mas2);
    pm = reinterpret_cast<uint8_t *>(mas);
  }

  k = 0ULL;
  if (mas[size - 1ULL] < 0.0) {
    for (i = size; i > 0ULL; i--) {
      if (mas[i - 1] > 0.0) {
        break;
      }
      mas2[k++] = mas[i - 1ULL];
    }

    for (i = 0ULL; i < size; i++) {
      if (mas[i] < 0.0) {
        break;
      }
      mas2[k++] = mas[i];
    }

    memcpy(mas, mas2, size * sizeof(double));
  }
}

bool ZagryadskovMRadixSortDoubleSimpleMergeSEQ::RunImpl() {
  GetOutput() = GetInput();
  RadixSortLSD(GetOutput().data(), GetOutput().size());

  return !GetOutput().empty();
}

bool ZagryadskovMRadixSortDoubleSimpleMergeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zagryadskov_m_radix_sort_double_simple_merge
