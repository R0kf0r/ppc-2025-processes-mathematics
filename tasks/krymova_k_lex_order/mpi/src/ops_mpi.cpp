#include "krymova_k_lex_order/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <string>
#include <vector>

#include "krymova_k_lex_order/common/include/common.hpp"

namespace krymova_k_lex_order {

KrymovaKLexOrderMPI::KrymovaKLexOrderMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KrymovaKLexOrderMPI::ValidationImpl() {
  return true;
}

bool KrymovaKLexOrderMPI::PreProcessingImpl() {
  return true;
}

bool KrymovaKLexOrderMPI::RunImpl() {
  const std::string &str1 = std::get<0>(GetInput());
  const std::string &str2 = std::get<1>(GetInput());

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int len1 = 0;
  int len2 = 0;
  if (rank == 0) {
    len1 = static_cast<int>(str1.length());
    len2 = static_cast<int>(str2.length());
  }

  MPI_Bcast(&len1, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&len2, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int min_len = std::min(len1, len2);
  int chunk_size = (min_len + size - 1) / size;
  int total_size = chunk_size * size;

  std::vector<char> sendbuf1(total_size, 0);
  std::vector<char> sendbuf2(total_size, 0);
  std::vector<char> local_str1(chunk_size, 0);
  std::vector<char> local_str2(chunk_size, 0);

  if (rank == 0) {
    std::copy(str1.begin(), str1.begin() + min_len, sendbuf1.begin());
    std::copy(str2.begin(), str2.begin() + min_len, sendbuf2.begin());
  }

  MPI_Scatter(sendbuf1.data(), chunk_size, MPI_CHAR, local_str1.data(), chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Scatter(sendbuf2.data(), chunk_size, MPI_CHAR, local_str2.data(), chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

  int start = rank * chunk_size;
  int end = std::min(start + chunk_size, min_len);
  int actual_size = end - start;

  int local_diff_pos = min_len;

  for (int i = 0; i < actual_size; ++i) {
    if (local_str1[i] != local_str2[i]) {
      local_diff_pos = start + i;
      break;
    }
  }

  int global_first_diff = min_len;
  MPI_Allreduce(&local_diff_pos, &global_first_diff, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  int result = 0;

  if (global_first_diff < min_len) {
    char char1 = 0;
    char char2 = 0;

    if (rank == 0) {
      char1 = str1[global_first_diff];
      char2 = str2[global_first_diff];
    }

    MPI_Bcast(&char1, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&char2, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    result = (char1 < char2) ? -1 : 1;
  } else {
    if (len1 < len2) {
      result = -1;
    } else if (len1 > len2) {
      result = 1;
    } else {
      result = 0;
    }
  }

  GetOutput() = result;
  return true;
}

bool KrymovaKLexOrderMPI::PostProcessingImpl() {
  return true;
}

}  // namespace krymova_k_lex_order
