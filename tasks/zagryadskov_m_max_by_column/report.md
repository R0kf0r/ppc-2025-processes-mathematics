## Министерство науки и высшего образования Российской Федерации  
## Федеральное государственное автономное образовательное учреждение высшего образования  
## **«Национальный исследовательский Нижегородский государственный университет им. Н.И. Лобачевского»**  
### Институт информационных технологий, математики и механики  

### **Направление подготовки:** «Прикладная математика и информатика»  

---

# Отчёт  
### По задаче  
**Нахождение максимальных значений по столбцам матрицы**  
**Вариант №16**

**Выполнил:**  
Студент группы 3823Б1ПМоп3  
**Загрядсков М.А.**

**Преподаватель:**  
доцент **Сысоев А.В.**

**Нижний Новгород, 2025**

---

## Введение

В данной работе реализован и исследован алгоритм нахождения максимальных значений по каждому столбцу матрицы. Задача решается в последовательной и в параллельной форме с использованием технологии MPI (Message Passing Interface). Параллельная реализация направлена на сокращение времени вычислений за счёт распределения данных по процессам.

---

## Постановка задачи

Необходимо для заданной матрицы, представленной в виде одномерного массива (вектор `std::vector<double>`), найти максимальные значения по каждому из её столбцов.  

Пусть матрица имеет размерность **m × n**, где  
- *m* — число строк,  
- *n* — число столбцов.  

Элементы матрицы хранятся **по столбцам**, то есть элемент с индексом *(i, j)* находится в массиве под индексом `j * m + i`.

Требуется:  
1. Реализовать последовательный алгоритм нахождения максимальных элементов по столбцам.  
2. Разработать параллельную реализацию с использованием **MPI**, распределяющую вычисления между несколькими процессами.  
3. Проверить корректность вычислений и провести экспериментальные замеры времени выполнения.  

---

## Описание алгоритма

**Последовательный алгоритм** выполняет следующие шаги:

1. Проверяет корректность входных данных.  
2. Инициализирует вектор результатов значением, минимально возможным для данного типа.  
3. Для каждого столбца `j`:
   - проходится по всем элементам строк `i`,
   - сравнивает текущее значение с максимумом столбца,
   - обновляет максимум при необходимости.  
4. Возвращает вектор максимальных значений по столбцам.

Псевдокод:  
```
for j in 0..n-1:
max_j = -∞ 
for i in 0..m-1:  
if A[i,j] > max_j:  
max_j = A[i,j]  
result[j] = max_j 
```

---

## Описание схемы параллельного алгоритма

Параллельная версия использует **распределение столбцов** между MPI-процессами.  

1. Корневой процесс (`rank 0`) вычисляет количество столбцов для обработки каждым процессом.
2. С корневого процесса блоки столбцов распределяются между процессами через `MPI_Scatterv`.  
3. Каждый процесс вычисляет локальные максимумы по выделенным столбцам.  
4. Локальные результаты собираются на корневом процессе с помощью `MPI_Gatherv`.  
5. Результат рассылается всем процессам через `MPI_Bcast` для обеспечения корректного прохождения функциональных тестов всеми процессами.  

Таким образом, каждый процесс работает с собственной частью данных, что позволяет достичь ускорения при достаточно больших размерах матриц.

---

## Описание MPI-версии

Программная реализация использует тип входных данных в виде кортежа:  
`std::tuple<size_t, std::vector<double>>`, 
где  
- первый элемент — количество столбцов `n`,  
- второй — вектор значений матрицы, хранящийся по столбцам.  

MPI-реализация опеределяет количество столбцов для каждого процесса, производит рассылку данных и сбор частичных результатов. Используется синхронизация через `MPI_Barrier`.

---

## Результаты экспериментов и подтверждение корректности

Эксперименты проводились на локальной машине.  
Параметры тестовой матрицы:
- размер **2048 × 4096**.
- элементы типа `double`.
- использовался генератор псевдослучайных чисел `mt19937`, также известный известный как **«Вихрь Мерсенна»**.
- Диапазон значений элементов матрицы: от -10<sup>6</sup> до 10<sup>6</sup>.


| Версия алгоритма | Время выполнения (с) |
|------------------:|---------------------:|
| Последовательная | 0.053 |
| Параллельная (MPI) | 0.019 |

**Подтверждение корректности:**  
Функция тестирования проверяет, что для каждого столбца найденный максимум не меньше всех элементов столбца.  
Все функциональные тесты и тесты производительности были успешно пройдены на локальной машине.

---

## Выводы из результатов

Реализация с использованием MPI показывает ускорение примерно в **2.8 раза** по сравнению с последовательной версией запуске на локальном устройстве.  
Это демонстрирует эффективность параллельного подхода при работе с крупными матрицами.  
При увеличении числа процессов можно ожидать дальнейшего сокращения времени выполнения, однако при малых размерах матриц затраты на коммуникацию могут нивелировать прирост производительности.

---

## Заключение

В работе реализованы и протестированы последовательная и параллельная (MPI) версии алгоритма нахождения максимальных значений по столбцам матрицы.  
Проведённые эксперименты подтвердили корректность и эффективность параллельной реализации.  

---

## Список литературы

1. Документация в формате веб-сайта по реализации **MPICH** стандарта **MPI**: [https://www.mpich.org](https://www.mpich.org)

---

## Приложение

### Параллельная реализация

```cpp
int world_size = 0;
  int world_rank = 0;
  int err_code = 0;
  err_code = MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_size failed");
  }
  err_code = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_rank failed");
  }
  int n = 0;
  const void *mat_data = nullptr;
  int m = 0;
  OutType &res = GetOutput();
  OutType local_res;
  OutType columns;
  std::vector<int> sendcounts(world_size);
  std::vector<int> displs(world_size);
  if (!displs.empty()) {
    displs[0] = 0;
  }

  if (world_rank == 0) {
    n = static_cast<int>(std::get<0>(GetInput()));
    const auto &mat = std::get<1>(GetInput());
    m = static_cast<int>(mat.size()) / n;
    mat_data = reinterpret_cast<const void *>(mat.data());
  }
  err_code = MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Bcast failed");
  }
  err_code = MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Bcast failed");
  }

  int columns_count = n / world_size;
  using T = double;  // datatype cannot be sent to other processes
  MPI_Datatype datatype = MPI_DOUBLE;

  int i = 0;
  int j = 0;
  int r = 0;
  T tmp = std::numeric_limits<T>::lowest();
  bool tmp_flag = false;

  if (world_rank == 0) {
    res.assign(n, std::numeric_limits<T>::lowest());
  }
  for (r = 0; r < world_size; ++r) {
    sendcounts[r] = (columns_count + static_cast<int>(r < (n % world_size))) * m;
    if (r > 0) {
      displs[r] = displs[r - 1] + sendcounts[r - 1];
    }
  }

  local_res.assign(static_cast<size_t>(sendcounts[world_rank] / m), std::numeric_limits<T>::lowest());
  columns.resize(sendcounts[world_rank]);
  err_code = MPI_Scatterv(mat_data, sendcounts.data(), displs.data(), datatype, columns.data(), sendcounts[world_rank],
                          datatype, 0, MPI_COMM_WORLD);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Scatterv failed");
  }
  for (j = 0; std::cmp_less(j, local_res.size()); ++j) {
    for (i = 0; i < m; ++i) {
      tmp = columns[(j * m) + i];
      tmp_flag = tmp > local_res[j];
      local_res[j] = (static_cast<T>(tmp_flag) * tmp) + (static_cast<T>(!tmp_flag) * local_res[j]);
    }
  }

  for (r = 0; r < world_size; ++r) {
    sendcounts[r] /= m;
    if (r > 0) {
      displs[r] = displs[r - 1] + sendcounts[r - 1];
    }
  }

  err_code = MPI_Gatherv(local_res.data(), static_cast<int>(local_res.size()), datatype, res.data(), sendcounts.data(),
                         displs.data(), datatype, 0, MPI_COMM_WORLD);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Gatherv failed");
  }
  if (world_rank != 0) {
    res.resize(n);
  }
  // sequential version requires not to call MPI funcs
  err_code = MPI_Bcast(res.data(), static_cast<int>(res.size()), datatype, 0, MPI_COMM_WORLD);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Bcast failed");
  }

  bool result = false;
  if (world_rank == 0) {
    result = !res.empty();
  } else {
    result = true;
  }
  err_code = MPI_Barrier(MPI_COMM_WORLD);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Barrier failed");
  }
  return result;
```