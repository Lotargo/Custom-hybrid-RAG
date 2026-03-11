cmake_minimum_required(VERSION 3.20)

project(SonataDB VERSION 0.1.0 LANGUAGES CXX)

# Включаем стандарт C++20 или C++23 (здесь используем C++20 как базу, можно C++23)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Глобальные опции
option(RAG_BUILD_TESTS "Build tests" ON)
option(RAG_BUILD_BENCHMARKS "Build benchmarks" ON)
option(RAG_ENABLE_SANITIZERS "Enable Address and UndefinedBehavior sanitizers" OFF)

# Настройка предупреждений компилятора
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -mavx2 -mfma)
endif()

if(RAG_ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

# Главная библиотека
add_library(sonatadb_core)
target_include_directories(sonatadb_core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
find_package(Threads REQUIRED)
target_link_libraries(sonatadb_core PRIVATE Threads::Threads)

# Подключаем модули
add_subdirectory(src)

if(RAG_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(RAG_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
