#include "zagryadskov_m_radix_sort_double_simple_merge/mpi/include/radix_sort_double_simple_merge.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "zagryadskov_m_radix_sort_double_simple_merge/common/include/common.hpp"
#include "zagryadskov_m_radix_sort_double_simple_merge/seq/include/radix_sort_double_simple_merge.hpp"

namespace zagryadskov_m_radix_sort_double_simple_merge {

ZagryadskovMRadixSortDoubleSimpleMergeMPI::ZagryadskovMRadixSortDoubleSimpleMergeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int world_rank = 0;
  int err_code = 0;
  err_code = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_rank failed");
  }
  if (world_rank == 0) {
    GetInput() = in;
  }
}

bool ZagryadskovMRadixSortDoubleSimpleMergeMPI::ValidationImpl() {
  bool res = false;
  int world_rank = 0;
  int err_code = 0;
  err_code = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_rank failed");
  }
  if (world_rank == 0) {
    res = !GetInput().empty();
  } else {
    res = true;
  }
  return res;
}

bool ZagryadskovMRadixSortDoubleSimpleMergeMPI::PreProcessingImpl() {
  return true;
}

void ZagryadskovMRadixSortDoubleSimpleMergeMPI::MyMPIMerge(std::vector<double> &data) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<double> tmp;

  for (int step = 1; step < size; step <<= 1) {
    if ((rank % (2 * step)) == step) {
      uint64_t n = data.size();
      MPI_Send(&n, 1, MPI_UINT64_T, rank - step, 0, MPI_COMM_WORLD);
      MPI_Send(data.data(), static_cast<int>(n), MPI_DOUBLE, rank - step, 1, MPI_COMM_WORLD);

      return;
    }

    int partner = rank + step;
    if (partner < size) {
      uint64_t recv_size = 0;
      MPI_Recv(&recv_size, 1, MPI_UINT64_T, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      tmp.resize(data.size() + recv_size);
      std::vector<double> recvbuf(recv_size);
      MPI_Recv(recvbuf.data(), static_cast<int>(recv_size), MPI_DOUBLE, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::ranges::merge(data.begin(), data.end(), recvbuf.begin(), recvbuf.end(), tmp.begin());
      data.swap(tmp);
    }
  }
}

bool ZagryadskovMRadixSortDoubleSimpleMergeMPI::RunImpl() {
  int world_size = 0;
  int world_rank = 0;
  int err_code = 0;
  bool res = true;
  err_code = MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_size failed");
  }
  err_code = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_rank failed");
  }
  size_t data_size = 0;
  auto world_size_st = static_cast<size_t>(world_size);
  std::vector<double> data;
  double *in_data = nullptr;
  std::vector<int> sendcounts(world_size_st);
  std::vector<int> displs(world_size_st);
  if (world_rank == 0) {
    data_size = GetInput().size();
    in_data = GetInput().data();
  }
  err_code = MPI_Bcast(&data_size, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Bcast failed");
  }
  size_t data_by_process = data_size / world_size_st;
  MPI_Datatype datatype = MPI_DOUBLE;

  for (size_t rk = 0; rk < world_size_st; ++rk) {
    sendcounts[rk] = static_cast<int>(data_by_process + static_cast<size_t>(rk < (data_size % world_size_st)));
    if (rk > 0) {
      displs[rk] = displs[rk - 1] + sendcounts[rk - 1];
    }
  }

  data.resize(sendcounts[world_rank]);
  err_code = MPI_Scatterv(in_data, sendcounts.data(), displs.data(), datatype, data.data(), sendcounts[world_rank],
                          datatype, 0, MPI_COMM_WORLD);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Scatterv failed");
  }

  ZagryadskovMRadixSortDoubleSimpleMergeSEQ::RadixSortLSD(data.data(), data.size());
  MyMPIMerge(data);

  if (world_rank == 0) {
    GetOutput() = data;
    res = !GetOutput().empty();
  } else {
    res = true;
  }
  return res;
}

bool ZagryadskovMRadixSortDoubleSimpleMergeMPI::PostProcessingImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int size = 0;
  if (world_rank == 0) {
    size = static_cast<int>(GetOutput().size());
  }
  MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput().resize(size);
  MPI_Bcast(GetOutput().data(), size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return !GetOutput().empty();
}

}  // namespace zagryadskov_m_radix_sort_double_simple_merge
