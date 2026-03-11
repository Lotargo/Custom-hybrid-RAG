#include "sonatadb/inference/thread_pool.hpp"
#include <iostream>

#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#endif

#if defined(__linux__)
#include <sched.h>
#elif defined(__APPLE__)
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#endif

namespace sonatadb::inference {

AffinityThreadPool::AffinityThreadPool(const std::vector<int>& core_ids)
    : stop_(false), affinity_success_(true) {

    std::latch setup_latch(core_ids.size());

    // Create threads and immediately set their affinity
    for (int core_id : core_ids) {
        threads_.emplace_back([this, core_id, &setup_latch]() {
            bool success = this->set_thread_affinity(core_id);
            if (!success) {
                affinity_success_.store(false, std::memory_order_relaxed);
            }
            setup_latch.count_down();
            this->worker_loop();
        });
    }

    // Wait until all threads have attempted to set their affinity
    setup_latch.wait();
}

AffinityThreadPool::~AffinityThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }

    // Wake up all threads so they can exit
    condition_.notify_all();

    for (std::thread& worker : threads_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void AffinityThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(this->queue_mutex_);
            // Wait until a task is available or the pool is stopped
            this->condition_.wait(lock, [this]() {
                return this->stop_ || !this->tasks_.empty();
            });

            if (this->stop_ && this->tasks_.empty()) {
                return; // Exit the thread
            }

            task = std::move(this->tasks_.front());
            this->tasks_.pop();
        }

        // Execute the task outside the lock
        task();
    }
}

bool AffinityThreadPool::set_thread_affinity(int core_id) {
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t current_thread = pthread_self();
    int result = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        std::cerr << "Failed to set thread affinity for core " << core_id << ": error code " << result << std::endl;
        return false;
    }
    return true;
#elif defined(__APPLE__)
    // macOS thread affinity API is different (thread policy).
    // It's a hint to the kernel to schedule threads on the same L2 cache if they share the same tag.
    // It does not pin to a specific logical core index like Linux does.
    // We provide a basic implementation to avoid errors but note it's less strict.

    thread_affinity_policy_data_t policy = { core_id };
    thread_port_t mach_thread = pthread_mach_thread_np(pthread_self());
    kern_return_t kr = thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY,
                                         (thread_policy_t)&policy, THREAD_AFFINITY_POLICY_COUNT);
    if (kr != KERN_SUCCESS) {
        std::cerr << "Failed to set thread affinity policy for core " << core_id << ": error code " << kr << std::endl;
        return false;
    }
    return true;
#else
    // Fallback for unsupported platforms (e.g., Windows without specific API calls)
    std::cerr << "Thread affinity is not supported on this platform. Ignoring core " << core_id << std::endl;
    return false;
#endif
}

} // namespace sonatadb::inference
