#include "krasnopevtseva_v_monte_carlo_integration/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <tuple>

#include "krasnopevtseva_v_monte_carlo_integration/common/include/common.hpp"

namespace krasnopevtseva_v_monte_carlo_integration {

KrasnopevtsevaVMCIntegrationSEQ::KrasnopevtsevaVMCIntegrationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KrasnopevtsevaVMCIntegrationSEQ::ValidationImpl() {
  const auto &input = GetInput();
  double a = std::get<0>(input);
  double b = std::get<1>(input);
  int num_points = std::get<2>(input);
  std::uint8_t func = std::get<3>(input);

  return (a <= b) && (num_points > 0) && (func <= 3);
}

bool KrasnopevtsevaVMCIntegrationSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool KrasnopevtsevaVMCIntegrationSEQ::RunImpl() {
  const auto &input = GetInput();
  double a = std::get<0>(input);
  double b = std::get<1>(input);
  int num_points = std::get<2>(input);
  std::uint8_t func = std::get<3>(input);

  double sum = 0.0;

  // генератор вихрь Мерсенна
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(a, b);

  // cos(x)*x^3
  for (int i = 0; i < num_points; i++) {
    double x = dis(gen);
    double fx = FuncSystem::GetFunc(func, x);
    sum += fx;
  }

  double integral = (b - a) * sum / num_points;
  GetOutput() = integral;

  return true;
}

bool KrasnopevtsevaVMCIntegrationSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace krasnopevtseva_v_monte_carlo_integration
