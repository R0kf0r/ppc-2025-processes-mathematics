#include "orehov_n_character_frequency/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "orehov_n_character_frequency/common/include/common.hpp"

namespace orehov_n_character_frequency {

OrehovNCharacterFrequencyMPI::OrehovNCharacterFrequencyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool OrehovNCharacterFrequencyMPI::ValidationImpl() {
  int rank = 0;
  int check = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    std::string text = std::get<0>(GetInput());
    std::string target_char = std::get<1>(GetInput());

    check = static_cast<int>((!text.empty()) && (target_char.length() == 1));
  }

  MPI_Bcast(&check, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return (check == 1);
}

bool OrehovNCharacterFrequencyMPI::PreProcessingImpl() {
  return true;
}

bool OrehovNCharacterFrequencyMPI::RunImpl() {
  int rank = 0;
  int size = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string local_str;
  std::string symbol;
  int length = 0;
  int global_result = 0;
  int local_result = 0;

  if (rank == 0) {
    std::string str = std::get<0>(GetInput());
    symbol = std::get<1>(GetInput());
    length = static_cast<int>(str.length());
    MPI_Bcast(symbol.data(), 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int part_size = length / size;
    int remains = length % size;

    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);

    for (int i = 0; i < size; i++) {
      sendcounts[i] = part_size + (i < remains ? 1 : 0);
      displs[i] = (i * part_size) + std::min(i, remains);
    }

    local_str.resize(sendcounts[0]);

    MPI_Scatterv(str.data(), sendcounts.data(), displs.data(), MPI_CHAR, local_str.data(), sendcounts[0], MPI_CHAR, 0,
                 MPI_COMM_WORLD);
  } else {
    symbol.resize(1);
    MPI_Bcast(symbol.data(), 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int part_size = length / size;
    int remains = length % size;

    int local_size = part_size + (rank < remains ? 1 : 0);
    local_str.resize(local_size);

    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_CHAR, local_str.data(), local_size, MPI_CHAR, 0, MPI_COMM_WORLD);
  }

  for (size_t i = 0; i < local_str.length(); i++) {
    if (local_str[i] == symbol[0]) {
      local_result++;
    }
  }

  MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = global_result;

  return true;
}

bool OrehovNCharacterFrequencyMPI::PostProcessingImpl() {
  return true;
}

}  // namespace orehov_n_character_frequency
