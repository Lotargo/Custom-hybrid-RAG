#include <gtest/gtest.h>
#include "rag_engine/inference/thread_pool.hpp"
#include <numeric>
#include <chrono>

using namespace rag_engine::inference;

TEST(AffinityThreadPoolTest, BasicExecution) {
    // Create a pool with 2 threads bound to cores 0 and 1 (if available).
    // Note: core 0 and 1 usually exist on most systems.
    // If affinity fails (e.g., inside a restricted container), it's fine, the tasks should still run.
    std::vector<int> cores = {0, 0}; // just testing logic, bind both to core 0 as a safe bet
    AffinityThreadPool pool(cores);

    EXPECT_EQ(pool.size(), 2);

    auto future1 = pool.submit([]() { return 42; });
    auto future2 = pool.submit([](int x, int y) { return x + y; }, 10, 20);

    EXPECT_EQ(future1.get(), 42);
    EXPECT_EQ(future2.get(), 30);
}

TEST(AffinityThreadPoolTest, StressTest) {
    std::vector<int> cores = {0, 0, 0, 0}; // 4 threads
    AffinityThreadPool pool(cores);

    const int num_tasks = 1000;
    std::vector<std::future<int>> futures;

    for (int i = 0; i < num_tasks; ++i) {
        futures.push_back(pool.submit([i]() { return i * 2; }));
    }

    int sum = 0;
    for (int i = 0; i < num_tasks; ++i) {
        sum += futures[i].get();
    }

    int expected_sum = 0;
    for (int i = 0; i < num_tasks; ++i) {
        expected_sum += i * 2;
    }

    EXPECT_EQ(sum, expected_sum);
}

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>

TEST(AffinityThreadPoolTest, CheckAffinity) {
    // Determine number of available cores
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
        GTEST_SKIP() << "Failed to get available CPU cores";
    }

    int available_core = -1;
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (CPU_ISSET(i, &cpuset)) {
            available_core = i;
            break;
        }
    }

    if (available_core == -1) {
         GTEST_SKIP() << "No available cores found to test affinity";
    }

    std::vector<int> cores = {available_core};
    AffinityThreadPool pool(cores);

    if (!pool.check_affinity_success()) {
        GTEST_SKIP() << "Affinity setting failed at pool initialization (maybe restricted environment)";
    }

    auto future = pool.submit([available_core]() {
        cpu_set_t task_cpuset;
        CPU_ZERO(&task_cpuset);
        pthread_t current_thread = pthread_self();
        int result = pthread_getaffinity_np(current_thread, sizeof(cpu_set_t), &task_cpuset);
        if (result != 0) {
            return false;
        }
        return CPU_ISSET(available_core, &task_cpuset) != 0;
    });

    EXPECT_TRUE(future.get());
}
#endif
