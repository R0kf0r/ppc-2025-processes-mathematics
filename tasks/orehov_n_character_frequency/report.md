# отчёт

 - задача 1
 - вариант 23
 - сделал: Орехов Никита Антонович
 - группа: 3823Б1ПМоп3

## 1. Introduction
Задача заключается в подсчёте частоты определённого символа в строке. 
Цель - реализовать последовательную и параллельную версию и сравнить их производительность.
Ожидание - выигрыш параллельной версии

## 2. Problem Statement
**Входные данные:**
 - Строка (Берётся из файла)
 - Целевой символ (Берётся из следующей строки в том же файле). 

**Выходные данные:**

 - Частота целевого символа в строке

**Ограничения**

 - Строка не должна быть пустой
 - Строка с целевым символом должна содержать ровно один символ

## 3. Baseline Algorithm (Sequential)

В цикле проверяем на совпадение поочередно каждый символ. 
При совпадении увеличиваем счётчик

## 4. Parallelization Scheme

Вначале считаем сколько данных получит каждый процесс. Распределяем их с помощью scatterv. 
После каждый процесс считает частоту символа в своей части данных
Дальше с помощью MPI_AllReduce собираем данные и складываем их
И результат отправляется всем процессам

**Распределение данных:**

Данные делятся на равные части между процессами. 
Оставшиеся данные распределяются равномерно между первыми процессами

**Шаблон коммуникаций**

Используется AllReduce для суммирования частот со всех процессов и отправки результата всем

**Роли процессов**

Все равны

## 5. Implementation Details

'''cpp
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
'''

## 6. Experimental Setup

 - Процессор: Процессор	Intel(R) Core(TM) i5-9300H CPU @ 2.40GHz   2.40 GHz
 - Количество ядер: 4
 - RAM: 16
 - OS Version: Windows 10


## 7. Performance

 Mode         | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| seq         | 1     | 0.04743 | 1.00    | N/A        |
| omp         | 4     | 0.05366 | 0.88    | 22%        |


## 8. Conclusions

Параллельная версия показала замедление. 
Основная причина в отсутствии вычислений, кроме счётчика. 
Время MPI коммуникаций значительно преобладает над временем вычислений

