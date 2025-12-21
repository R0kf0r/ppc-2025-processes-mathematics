#include "zagryadskov_m_allreduce/seq/include/allreduce.hpp"

#include <mpi.h>

#include <cstddef>
#include <stdexcept>

#include "zagryadskov_m_allreduce/common/include/common.hpp"

namespace zagryadskov_m_allreduce {

ZagryadskovMAllreduceSEQ::ZagryadskovMAllreduceSEQ(const InType &in) {
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

bool ZagryadskovMAllreduceSEQ::ValidationImpl() {
  bool res = false;
  int world_rank = 0;
  int world_size = 0;
  int err_code = 0;
  err_code = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_rank failed");
  }
  err_code = MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_rank failed");
  }
  if (world_rank == 0) {
    auto &param1 = std::get<0>(GetInput());
    int param2 = std::get<1>(GetInput());
    int param3 = std::get<2>(GetInput());

    res = (!param1.empty()) && (param3 >= 0) && (param3 <= 2) && (param2 > 0) &&
          (param1.size() >= static_cast<size_t>(param2) * static_cast<size_t>(world_size));
  } else {
    res = true;
  }
  return res;
}

bool ZagryadskovMAllreduceSEQ::PreProcessingImpl() {
  int world_rank = 0;
  int err_code = 0;
  int count = 0;
  err_code = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_rank failed");
  }
  if (world_rank == 0) {
    count = std::get<1>(GetInput());
  }

  err_code = MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Bcast failed");
  }
  temp_vec_.resize(count);

  err_code =
      MPI_Scatter(std::get<0>(GetInput()).data(), count, MPI_INT, temp_vec_.data(), count, MPI_INT, 0, MPI_COMM_WORLD);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Scatter failed");
  }

  err_code = MPI_Barrier(MPI_COMM_WORLD);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Barrier failed");
  }
  return true;
}

MPI_Op ZagryadskovMAllreduceSEQ::GetOp(int iop) {
  MPI_Op op = MPI_OP_NULL;
  switch (iop) {
    case 0:
      op = MPI_MAX;
      break;
    case 1:
      op = MPI_MIN;
      break;
    case 2:
      op = MPI_SUM;
      break;
    default:
      op = MPI_OP_NULL;
      break;
  }

  return op;
}

bool ZagryadskovMAllreduceSEQ::RunImpl() {
  int world_size = 0;
  int world_rank = 0;
  int err_code = 0;
  int iop = 0;
  err_code = MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_size failed");
  }
  err_code = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_rank failed");
  }
  if (world_rank == 0) {
    iop = std::get<2>(GetInput());
  }
  err_code = MPI_Bcast(&iop, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Bcast failed");
  }

  GetOutput().resize(temp_vec_.size());
  MPI_Op op = ZagryadskovMAllreduceSEQ::GetOp(iop);
  MPI_Allreduce(temp_vec_.data(), GetOutput().data(), static_cast<int>(temp_vec_.size()), MPI_INT, op, MPI_COMM_WORLD);

  err_code = MPI_Barrier(MPI_COMM_WORLD);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Barrier failed");
  }
  return true;
}

bool ZagryadskovMAllreduceSEQ::PostProcessingImpl() {
  bool result = false;
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank == 0) {
    result = !GetOutput().empty();
  } else {
    result = true;
  }
  return result;
}

}  // namespace zagryadskov_m_allreduce
