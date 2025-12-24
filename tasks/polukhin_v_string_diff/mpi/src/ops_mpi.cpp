#include "polukhin_v_string_diff/mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "mpi.h"
#include "polukhin_v_string_diff/common/include/common.hpp"

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
bool StringDiffTaskMPI::PostProcessingImpl() {
  return true;
}

namespace {

void BroadcastStringLengths(int rank, int &len1, int &len2) {
  if (rank == 0) {
  }
  MPI_Bcast(&len1, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&len2, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void PrepareDataDistribution(int rank, int size, int min_len, std::vector<int> &sendcounts, std::vector<int> &displs) {
  if (rank != 0) {
    return;
  }

  std::ranges::fill(sendcounts, 0);
  std::ranges::fill(displs, 0);

  const int els_per_process = (min_len + size - 1) / size;
  int offset = 0;

  for (int i = 0; i < size; ++i) {
    const int start = i * els_per_process;
    const int end = std::min(start + els_per_process, min_len);

    if (start < min_len) {
      sendcounts[i] = end - start;
    }

    displs[i] = offset;

    if (start < min_len) {
      offset += sendcounts[i];
    }
  }
}

uint64_t ComputeLocalDifferences(const std::vector<char> &local_str1, const std::vector<char> &local_str2) {
  uint64_t local_count = 0;
  const size_t recvcount = local_str1.size();

  for (size_t i = 0; i < recvcount; ++i) {
    if (local_str1[i] != local_str2[i]) {
      ++local_count;
    }
  }
  return local_count;
}

void DistributeStrings(int rank, int recvcount, const std::vector<int> &sendcounts, const std::vector<int> &displs,
                       const std::string &str1, const std::string &str2, std::vector<char> &local_str1,
                       std::vector<char> &local_str2) {
  if (recvcount <= 0) {
    return;
  }

  MPI_Scatterv(rank == 0 ? str1.data() : nullptr, sendcounts.data(), displs.data(), MPI_CHAR, local_str1.data(),
               recvcount, MPI_CHAR, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? str2.data() : nullptr, sendcounts.data(), displs.data(), MPI_CHAR, local_str2.data(),
               recvcount, MPI_CHAR, 0, MPI_COMM_WORLD);
}

void CollectAndSetResults(int rank, uint64_t local_count, int length_diff, size_t &output) {
  uint64_t total_count = 0;
  MPI_Reduce(&local_count, &total_count, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    total_count += static_cast<uint64_t>(length_diff);
    output = static_cast<size_t>(total_count);
  }

  uint64_t result = 0;
  if (rank == 0) {
    result = total_count;
  }
  MPI_Bcast(&result, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    output = static_cast<size_t>(result);
  }
}

}  // namespace

bool StringDiffTaskMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string str1;
  std::string str2;
  int len1 = 0;
  int len2 = 0;

  if (rank == 0) {
    const auto &input = GetInput();
    str1 = input.first;
    str2 = input.second;
    len1 = static_cast<int>(str1.size());
    len2 = static_cast<int>(str2.size());
  }

  BroadcastStringLengths(rank, len1, len2);

  const int min_len = std::min(len1, len2);
  const int length_diff = std::abs(len1 - len2);

  uint64_t local_count = 0;

  if (min_len > 0) {
    std::vector<int> sendcounts(size, 0);
    std::vector<int> displs(size, 0);

    PrepareDataDistribution(rank, size, min_len, sendcounts, displs);

    int recvcount = 0;
    MPI_Scatter(sendcounts.data(), 1, MPI_INT, &recvcount, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (recvcount > 0) {
      std::vector<char> local_str1(recvcount);
      std::vector<char> local_str2(recvcount);

      DistributeStrings(rank, recvcount, sendcounts, displs, str1, str2, local_str1, local_str2);

      local_count = ComputeLocalDifferences(local_str1, local_str2);
    }
  }

  CollectAndSetResults(rank, local_count, length_diff, GetOutput());

  return true;
}

}  // namespace polukhin_v_string_diff
