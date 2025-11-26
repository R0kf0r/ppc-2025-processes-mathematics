#include "polukhin_v_string_diff/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>

namespace polukhin_v_string_diff {

StringDiffTaskMPI::StringDiffTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool StringDiffTaskMPI::ValidationImpl() {
  return true;
}

bool StringDiffTaskMPI::PreProcessingImpl() {
  return true;
}

bool StringDiffTaskMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  const std::string &str1 = input.first;
  const std::string &str2 = input.second;

  int minLen = static_cast<int>(std::min(str1.size(), str2.size()));
  int lengthDiff = static_cast<int>(std::abs(static_cast<int>(str1.size()) - static_cast<int>(str2.size())));

  int localCount = 0;

  if (minLen > 0) {
    int calcNum = (minLen + size - 1) / size;
    int start = rank * calcNum;
    int end = std::min(start + calcNum, minLen);

    for (int i = start; i < end; ++i) {
      if (str1[i] != str2[i]) {
        ++localCount;
      }
    }
  }

  int totalCount = 0;
  MPI_Reduce(&localCount, &totalCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    totalCount += lengthDiff;
    GetOutput() = static_cast<size_t>(totalCount);
  }

  size_t result = (rank == 0) ? GetOutput() : 0;
  MPI_Bcast(&result, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput() = result;
  }

  return true;
}

bool StringDiffTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace polukhin_v_string_diff
