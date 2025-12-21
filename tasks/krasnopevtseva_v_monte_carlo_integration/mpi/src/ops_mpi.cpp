#include "krasnopevtseva_v_monte_carlo_integration/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <tuple>

#include "krasnopevtseva_v_monte_carlo_integration/common/include/common.hpp"

namespace krasnopevtseva_v_monte_carlo_integration {

KrasnopevtsevaVMCIntegrationMPI::KrasnopevtsevaVMCIntegrationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KrasnopevtsevaVMCIntegrationMPI::ValidationImpl() {
  const auto &input = GetInput();
  double a = std::get<0>(input);
  double b = std::get<1>(input);
  int num_points = std::get<2>(input);
  std::uint8_t func = std::get<3>(input);

  return (a <= b) && (num_points > 0) && (func <= 3);
}

bool KrasnopevtsevaVMCIntegrationMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool KrasnopevtsevaVMCIntegrationMPI::RunImpl() {
  const auto &input = GetInput();
  double a = std::get<0>(input);
  double b = std::get<1>(input);
  int num_points = std::get<2>(input);
  std::uint8_t func = std::get<3>(input);

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int local_points = num_points / size;
  int remainder = num_points % size;

  if (rank < remainder) {
    local_points++;
  }
  double local_sum = 0.0;
  std::mt19937 gen(std::random_device{}() + rank);
  std::uniform_real_distribution<double> dis(a, b);

  for (int i = 0; i < local_points; i++) {
    double x = dis(gen);
    double fx = FuncSystem::GetFunc(func, x);
    local_sum += fx;
  }
  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  double integral = 0.0;
  if (rank == 0) {
    integral = (b - a) * global_sum / num_points;
  }
  MPI_Bcast(&integral, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = integral;
  return true;
}

bool KrasnopevtsevaVMCIntegrationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace krasnopevtseva_v_monte_carlo_integration
