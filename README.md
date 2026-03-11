<div align="center">

![Custom RAG Hero Banner](./assets/hero-banner.svg)

![C++ Version](https://img.shields.io/badge/C%2B%2B-23-blue.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Performance](https://img.shields.io/badge/Performance-Zero--Allocation-orange.svg)

**Custom Hybrid SonataDB** — сверхбыстрый, распределенный гипер-масштабируемый движок для поиска (Vector + Full-text), построенный на принципах максимального контроля над железом и ОС. Проект ориентирован на использование в облачных инфраструктурах, HPC и на портативных устройствах (ARM).

</div>

---

<table>
<tr>
<td>

## <img src="./assets/architecture.svg" width="24" height="24" /> Философия и Архитектурное Видение

В отличие от популярных решений на Python или Go, этот движок строится "от железа вверх". Мы не просто используем библиотеки; мы переосмысляем то, как данные перемещаются между диском, кэшем процессора и регистрами SIMD.

### Ключевые принципы:
1. **Zero-Allocation на горячих путях:** Все структуры данных (индексы, векторы) маппируются напрямую в память (`mmap`), минимизируя работу аллокатора и сборщика мусора.
2. **Instruction-Level SLА:** Гарантия производительности через аппаратное разделение ресурсов (Intel RDT / ARM MPAM).
3. **Kernel-Bypass I/O:** Использование `io_uring` и `SPDK` для исключения оверхеда ядра при работе с NVMe.
4. **Портативность без компромиссов:** Поддержка x86_64 (AVX-512) и ARM (NEON/MPAM) через абстрактный слой диспетчеризации.

</td>
</tr>
</table>

---

## <img src="./assets/tech_stack.svg" width="24" height="24" /> Технологический Стек

- **Язык:** Modern C++ (C++23)
- **Сборка:** CMake 3.20+, Ninja
- **Ускорение:** AVX2, AVX-512, NEON (Runtime dispatch)
- **Хранилище:** Custom `mmap`-based engine, Write-Ahead Log (WAL), RocksDB (для метаданных)
- **Тестирование:** Google Test, Google Benchmark, Intel SDE (для эмуляции AVX-512)

---

<table>
<tr>
<td>

## <img src="./assets/patterns.svg" width="24" height="24" /> Ключевые Архитектурные Паттерны

Подробное описание всех паттернов находится в [patterns.md](./patterns.md).

### 1. Векторный индекс (HNSW)
Наша реализация HNSW спроектирована для работы в `mmap`-режиме:
- **Flat Memory Layout:** Граф хранится в плоских массивах внутри маппированного файла.
- **Lock-Free Search:** Поиск не блокируется при вставке новых элементов.
- **Fine-Grained Node Locking:** При обновлении связей используются спинлоки на уровне узлов.

### 2. Tiered Storage & Reliability
- **mmap + madvise:** Мы подсказываем ядру ОС, как работать с памятью (последовательно или рандомно), минимизируя Page Faults.
- **WAL (Write-Ahead Log):** Гарантия сохранности данных даже при жестком падении процесса.
- **LSM-style Compaction:** Удаление данных через Tombstones с последующим фоновым сжатием индекса.

### 3. SIMD Abstraction Layer
Универсальный слой для вычисления метрик (L2, Cosine, Dot), автоматически выбирающий лучший набор инструкций (AVX-512 > AVX2 > Scalar) во время выполнения.

</td>
</tr>
</table>

---

## Структура проекта

- **`include/sonatadb/math/`**: Математическое ядро, SIMD-интринсики и диспетчеризация.
- **`include/sonatadb/storage/`**: Низкоуровневая работа с диском (`mmap`), WAL и хранилище метаданных.
- **`include/sonatadb/index/`**: Реализация поисковых индексов (HNSW).
- **`src/`**: Реализация всех модулей.
- **`tests/`**: Юнит-тесты (GTest).
- **`benchmarks/`**: Бенчмарки производительности (Google Benchmark).

---

<table>
<tr>
<td>

## <img src="./assets/agents.svg" width="24" height="24" /> Кодекс работы AI-агентов

Этот проект содержит строгие инструкции для AI-агентов (см. [AGENTS.md](./AGENTS.md)). Основные правила:
1. **No Stubs:** Никаких заглушек или "TODO". Только рабочий код.
2. **Verifiability:** Любой новый модуль обязательно должен сопровождаться бенчмарком.
3. **No Debt:** Все решения должны быть окончательными и оптимизированными.

</td>
</tr>
</table>

---

## <img src="./assets/setup.svg" width="24" height="24" /> Сборка и Тестирование

### Зависимости:
- C++23 Compiler (GCC 13+, Clang 16+)
- CMake 3.20+
- Ninja (опционально)
- RocksDB (для модуля метаданных)

### Инструкция по сборке:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Запуск тестов:

```bash
# Обычный запуск (AVX-512 может скипаться, если CPU не поддерживает)
./build/tests/sonatadb_tests

# Запуск с эмуляцией AVX-512 (через Intel SDE)
/path/to/sde64 -skx -- ./build/tests/sonatadb_tests
```

### Запуск бенчмарков:

```bash
./build/benchmarks/sonatadb_benchmarks
```

---

## Дорожная карта (Roadmap)
Текущий прогресс и будущие задачи описаны в [TODO.md](./TODO.md) и [roadmap.md](./roadmap.md).
