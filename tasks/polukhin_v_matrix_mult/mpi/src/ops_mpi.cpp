#include "polukhin_v_matrix_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "polukhin_v_matrix_mult/common/include/common.hpp"

namespace polukhin_v_matrix_mult {

MatrixMultTaskMPI::MatrixMultTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool MatrixMultTaskMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Только процесс 0 имеет данные на входе
  if (rank == 0) {
    const auto &input = GetInput();
    const auto &dims = input.dims;

    if (dims.rows_a == 0 || dims.cols_a == 0 || dims.cols_b == 0) {
      return false;
    }

    size_t expected_size_a = dims.rows_a * dims.cols_a;
    size_t expected_size_b = dims.cols_a * dims.cols_b;

    if (input.matrix_a.size() != expected_size_a) {
      return false;
    }

    if (input.matrix_b.size() != expected_size_b) {
      return false;
    }
  }

  return true;
}

bool MatrixMultTaskMPI::PreProcessingImpl() {
  return true;
}

bool MatrixMultTaskMPI::PostProcessingImpl() {
  return true;
}

bool MatrixMultTaskMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // размеры
  int m = 0;  // строки A
  int k = 0;  // столбцы A = строки B
  int n = 0;  // столбцы B

  std::vector<double> a_matrix;
  std::vector<double> b_matrix;

  // процесс 0 берет входные данные
  if (rank == 0) {
    const auto &input = GetInput();
    a_matrix = input.matrix_a;
    b_matrix = input.matrix_b;
    m = static_cast<int>(input.dims.rows_a);
    k = static_cast<int>(input.dims.cols_a);
    n = static_cast<int>(input.dims.cols_b);
  }

  // рассылаем размеры всем процессам
  int dims[3];
  if (rank == 0) {
    dims[0] = m;
    dims[1] = k;
    dims[2] = n;
  }
  MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    m = dims[0];
    k = dims[1];
    n = dims[2];
  }

  // рассылаем матрицу B всем процессам
  int b_size = k * n;
  if (rank != 0) {
    b_matrix.resize(b_size);
  }
  MPI_Bcast(b_matrix.data(), b_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // сколько строк получит каждый процесс
  int base_rows = m / size;
  int extra_rows = m % size;

  // для каждого процесса вычисляем количество строк
  int my_rows = base_rows;

  if (rank < extra_rows) {
    my_rows = base_rows + 1;
  } else {
    my_rows = base_rows;
  }

  // подготовка параметров для Scatterv
  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);

  if (rank == 0) {
    // заполнение массивы sendcounts и displs, для всех процессов
    for (int i = 0; i < size; ++i) {
      int rows_for_proc = base_rows;
      int start_for_proc = 0;

      if (i < extra_rows) {
        rows_for_proc = base_rows + 1;
        start_for_proc = i * (base_rows + 1);
      } else {
        rows_for_proc = base_rows;
        start_for_proc = extra_rows * (base_rows + 1) + (i - extra_rows) * base_rows;
      }

      sendcounts[i] = rows_for_proc * k;
      displs[i] = start_for_proc * k;
    }
  }

  // буфер для локальных строк матрицы A
  std::vector<double> local_a(my_rows * k);

  // распределение строк матрицы A между процессами
  int my_count = my_rows * k;
  MPI_Scatterv(rank == 0 ? a_matrix.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE, local_a.data(),
               my_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // локальное умножение
  std::vector<double> local_result(my_rows * n);

  for (int i = 0; i < my_rows; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      // скалярное произведение
      for (int p = 0; p < k; ++p) {
        double val_a = local_a[i * k + p];
        double val_b = b_matrix[p * n + j];
        sum += val_a * val_b;
      }
      local_result[i * n + j] = sum;
    }
  }

  // сборка результатов обратно на процесс 0
  std::vector<int> recvcounts(size);
  std::vector<int> rdispls(size);

  if (rank == 0) {
    // вычисление параметры для Gatherv
    for (int i = 0; i < size; ++i) {
      int rows_for_proc = base_rows;
      int start_for_proc = 0;

      if (i < extra_rows) {
        rows_for_proc = base_rows + 1;
        start_for_proc = i * (base_rows + 1);
      } else {
        rows_for_proc = base_rows;
        start_for_proc = extra_rows * (base_rows + 1) + (i - extra_rows) * base_rows;
      }

      recvcounts[i] = rows_for_proc * n;
      rdispls[i] = start_for_proc * n;
    }

    // память под результат
    GetOutput().resize(m * n);
  }

  int my_result_count = my_rows * n;
  MPI_Gatherv(local_result.data(), my_result_count, MPI_DOUBLE, rank == 0 ? GetOutput().data() : nullptr,
              recvcounts.data(), rdispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (size > 1) {
    int total_result_size = m * n;
    if (rank != 0) {
      GetOutput().resize(total_result_size);
    }
    MPI_Bcast(GetOutput().data(), total_result_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  return true;
}

}  // namespace polukhin_v_matrix_mult
