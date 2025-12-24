#include "boltenkov_s_max_in_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <limits>
#include <vector>

#include "boltenkov_s_max_in_matrix/common/include/common.hpp"

namespace boltenkov_s_max_in_matrix {

BoltenkovSMaxInMatrixkMPI::BoltenkovSMaxInMatrixkMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  } else {
    GetInput() = InType{};
  }
  GetOutput() = -std::numeric_limits<double>::max();
}

bool BoltenkovSMaxInMatrixkMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return std::get<0>(GetInput()) > 0 && !std::get<1>(GetInput()).empty() &&
           std::get<1>(GetInput()).size() % std::get<0>(GetInput()) == 0;
  }
  return true;
}

bool BoltenkovSMaxInMatrixkMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return std::get<0>(GetInput()) > 0 && !std::get<1>(GetInput()).empty() &&
           std::get<1>(GetInput()).size() % std::get<0>(GetInput()) == 0;
  }
  return true;
}

bool BoltenkovSMaxInMatrixkMPI::RunImpl() {
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> sendcounts(size, 0);
  std::vector<int> displs(size, 0);
  std::vector<double> data;
  std::vector<double> all_maxs(size, std::numeric_limits<double>::lowest());

  OutType &mx = GetOutput();
  std::vector<double> &v = std::get<1>(GetInput());

  MPI_Datatype datatype = MPI_DOUBLE;

  int len = static_cast<int>(v.size());
  MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int cur_disp = 0;
  int cnt_item = len / size;
  int r = len % size;
  for (int i = 0; i < size; ++i) {
    int cur_cnt = cnt_item + (i < r ? 1 : 0);
    sendcounts[i] = cur_cnt;
    displs[i] = cur_disp;
    cur_disp += sendcounts[i];
  }
  data.resize(sendcounts[rank]);

  MPI_Scatterv((rank == 0) ? v.data() : nullptr, sendcounts.data(), displs.data(), datatype, data.data(),
               sendcounts[rank], datatype, 0, MPI_COMM_WORLD);

  bool flag = false;
  OutType tmp_mx = std::numeric_limits<double>::lowest();
  for (int i = 0; i < sendcounts[rank]; ++i) {
    flag = data[i] > tmp_mx;
    tmp_mx = (static_cast<double>(flag) * data[i]) + (static_cast<double>(!flag) * tmp_mx);
  }

  for (int i = 0; i < size; ++i) {
    sendcounts[i] = 1;
    displs[i] = i;
  }

  MPI_Gatherv(&tmp_mx, 1, datatype, all_maxs.data(), sendcounts.data(), displs.data(), datatype, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      flag = all_maxs[i] > mx;
      mx = (static_cast<double>(flag) * all_maxs[i]) + (static_cast<double>(!flag) * mx);
    }
  }

  MPI_Bcast(&mx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool BoltenkovSMaxInMatrixkMPI::PostProcessingImpl() {
  return true;
}

}  // namespace boltenkov_s_max_in_matrix
