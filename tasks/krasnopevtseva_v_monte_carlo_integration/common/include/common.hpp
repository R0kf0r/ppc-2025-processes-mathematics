#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace krasnopevtseva_v_monte_carlo_integration {

using InType = std::tuple<double, double, int, std::uint8_t>;
using OutType = double;
using TestType = std::tuple<std::tuple<double, double, int, std::uint8_t>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

enum class FuncIndex : std::uint8_t {
  kCosX3 = 0,  // cos(x) * x^3
  kSinX2 = 1,  // sin(x) * x^2
  kExpX = 2,   // exp(-x) * x
  kPolyX = 3   // x^4 - 2*x^2 + 1
};

class FuncSystem {
 public:
  static double GetFunc(std::uint8_t index, double x) {
    if (index == 0) {
      return std::cos(x) * x * x * x;
    }
    if (index == 1) {
      return std::sin(x) * x * x;
    }
    if (index == 2) {
      return std::exp(-x) * x;
    }
    return (x * x * x * x) - (2 * x * x) + 1;
  }

  static double AnalyticIntegral(std::uint8_t index, double a, double b) {
    if (index == 0) {
      return AnalyticCosX3(a, b);
    }
    if (index == 1) {
      return AnalyticSinX2(a, b);
    }
    if (index == 2) {
      return AnalyticExpX(a, b);
    }
    return AnalyticPolyX(a, b);
  }

 private:
  static double FCos(double x) {
    return ((x * x * x - 6 * x) * std::sin(x)) + ((3 * x * x - 6) * std::cos(x));
  }
  static double AnalyticCosX3(double a, double b) {
    return FCos(b) - FCos(a);
  }

  static double FSin(double x) {
    return (2 * std::sin(x) * x) - ((x * x - 2) * std::cos(x));
  }

  static double AnalyticSinX2(double a, double b) {
    return FSin(b) - FSin(a);
  }

  static double FExp(double x) {
    return -(x + 1) * std::exp(-x);
  }
  static double AnalyticExpX(double a, double b) {
    return FExp(b) - FExp(a);
  }

  static double FPoly(double x) {
    return ((x * x * x * x * x) / 5.0) - ((2 * x * x * x) / 3.0) + x;
  }

  static double AnalyticPolyX(double a, double b) {
    return FPoly(b) - FPoly(a);
  }
};

}  // namespace krasnopevtseva_v_monte_carlo_integration
