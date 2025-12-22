#include "polukhin_v_matrix_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
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

    const std::size_t expected_size_a = static_cast<std::size_t>(dims.rows_a) * static_cast<std::size_t>(dims.cols_a);
    const std::size_t expected_size_b = static_cast<std::size_t>(dims.cols_a) * static_cast<std::size_t>(dims.cols_b);

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

  // рассылаем размеры всем процессам (используем std::array вместо C-массива)
  std::array<int, 3> dims{};
  if (rank == 0) {
    dims[0] = m;
    dims[1] = k;
    dims[2] = n;
  }
  MPI_Bcast(dims.data(), static_cast<int>(dims.size()), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    m = dims[0];
    k = dims[1];
    n = dims[2];
  }

  // рассылаем матрицу B всем процессам
  const int b_size = k * n;
  if (rank != 0) {
    b_matrix.resize(static_cast<std::size_t>(b_size));
  }
  // если b_size == 0, MPI_Bcast корректно вызовется с nullptr на ранках-не-0? безопаснее передавать .data()
  MPI_Bcast(b_matrix.data(), b_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // сколько строк получит каждый процесс
  const int base_rows = (size == 0) ? 0 : (m / size);
  const int extra_rows = (size == 0) ? 0 : (m % size);

  // сколько строк у текущего процесса (вычисляем корректно, без "dead store")
  int my_rows = (rank < extra_rows) ? (base_rows + 1) : base_rows;

  // подготовка параметров для Scatterv
  std::vector<int> sendcounts(static_cast<std::size_t>(size));
  std::vector<int> displs(static_cast<std::size_t>(size));

  // лямбда для вычисления rows/start для процесса i (чтобы не дублировать код и уменьшить cognitive complexity)
  auto compute_rows_start = [&](int proc_index) -> std::pair<int, int> {
    if (proc_index < extra_rows) {
      const int rows_for_proc = base_rows + 1;
      const int start_for_proc = proc_index * (base_rows + 1);
      return {rows_for_proc, start_for_proc};
    }
    const int rows_for_proc = base_rows;
    const int start_for_proc = extra_rows * (base_rows + 1) + (proc_index - extra_rows) * base_rows;
    return {rows_for_proc, start_for_proc};
  };

  if (rank == 0) {
    // заполнение массивов sendcounts и displs, для всех процессов
    for (int i = 0; i < size; ++i) {
      auto pr = compute_rows_start(i);
      const int rows_for_proc = pr.first;
      const int start_for_proc = pr.second;

      sendcounts[static_cast<std::size_t>(i)] = rows_for_proc * k;
      displs[static_cast<std::size_t>(i)] = start_for_proc * k;
    }
  }

  // буфер для локальных строк матрицы A
  const std::size_t my_rows_sz = static_cast<std::size_t>(my_rows);
  const std::size_t k_sz = static_cast<std::size_t>(k);
  const std::size_t n_sz = static_cast<std::size_t>(n);

  std::vector<double> local_a(my_rows_sz * k_sz);

  // распределение строк матрицы A между процессами
  const int my_count = my_rows * k;
  MPI_Scatterv(rank == 0 ? a_matrix.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE, local_a.data(),
               my_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // локальное умножение
  std::vector<double> local_result(my_rows_sz * n_sz);

  for (int i = 0; i < my_rows; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      // скалярное произведение
      for (int pk = 0; pk < k; ++pk) {
        const double val_a = local_a[(static_cast<std::size_t>(i) * k_sz) + static_cast<std::size_t>(pk)];
        const double val_b = b_matrix[(static_cast<std::size_t>(pk) * n_sz) + static_cast<std::size_t>(j)];
        sum += val_a * val_b;
      }
      local_result[(static_cast<std::size_t>(i) * n_sz) + static_cast<std::size_t>(j)] = sum;
    }
  }

  // сборка результатов обратно на процесс 0
  std::vector<int> recvcounts(static_cast<std::size_t>(size));
  std::vector<int> rdispls(static_cast<std::size_t>(size));

  if (rank == 0) {
    // вычисление параметров для Gatherv
    for (int i = 0; i < size; ++i) {
      auto pr = compute_rows_start(i);
      const int rows_for_proc = pr.first;
      const int start_for_proc = pr.second;

      recvcounts[static_cast<std::size_t>(i)] = rows_for_proc * n;
      rdispls[static_cast<std::size_t>(i)] = start_for_proc * n;
    }

    // память под результат (size_t)
    GetOutput().resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
  }

  const int my_result_count = my_rows * n;
  MPI_Gatherv(local_result.data(), my_result_count, MPI_DOUBLE, rank == 0 ? GetOutput().data() : nullptr,
              recvcounts.data(), rdispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (size > 1) {
    const int total_result_size = m * n;
    if (rank != 0) {
      GetOutput().resize(static_cast<std::size_t>(total_result_size));
    }
    MPI_Bcast(GetOutput().data(), total_result_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  return true;
}

}  // namespace polukhin_v_matrix_mult
