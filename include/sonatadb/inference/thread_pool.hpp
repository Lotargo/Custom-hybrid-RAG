#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <optional>
#include <stdexcept>
#include <concepts>
#include <type_traits>
#include <memory>
#include <latch>

namespace sonatadb::inference {

/**
 * @brief Thread pool with thread affinity support.
 *
 * Optimized for ML inference tasks where setting thread affinity to specific
 * CPU cores prevents L1/L2 cache misses caused by the OS scheduler migrating
 * threads between cores. Uses standard C++ threads and POSIX pthreads for affinity.
 */
class AffinityThreadPool {
public:
    /**
     * @brief Construct a new Affinity Thread Pool.
     *
     * @param core_ids A list of logical core IDs to bind the threads to.
     *                 The number of threads created will equal the size of core_ids.
     *                 If a core ID is not valid or affinity fails, an exception is thrown.
     */
    explicit AffinityThreadPool(const std::vector<int>& core_ids);

    /**
     * @brief Destroy the Affinity Thread Pool, waiting for all currently
     *        enqueued tasks to finish.
     */
    ~AffinityThreadPool();

    // Non-copyable, non-movable
    AffinityThreadPool(const AffinityThreadPool&) = delete;
    AffinityThreadPool& operator=(const AffinityThreadPool&) = delete;
    AffinityThreadPool(AffinityThreadPool&&) = delete;
    AffinityThreadPool& operator=(AffinityThreadPool&&) = delete;

    /**
     * @brief Enqueues a task into the thread pool.
     *
     * @tparam F Type of the function/callable.
     * @tparam Args Types of the arguments to the function.
     * @param f Function to execute.
     * @param args Arguments to pass to the function.
     * @return std::future holding the return value of the function.
     */
    template <typename F, typename... Args>
        requires std::invocable<F, Args...>
    [[nodiscard]] auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using return_type = std::invoke_result_t<F, Args...>;

        // Package the task so it can be stored and its return value extracted via future
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            [f = std::forward<F>(f), ...args = std::forward<Args>(args)]() mutable {
                return std::invoke(f, args...);
            }
        );

        std::future<return_type> res = task->get_future();

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // Don't allow enqueuing after stopping the pool
            if (stop_) {
                throw std::runtime_error("enqueue on stopped AffinityThreadPool");
            }

            // Wrap the packaged task in a void function and push it to the queue
            tasks_.emplace([task]() { (*task)(); });
        }

        // Wake up one thread to execute the task
        condition_.notify_one();
        return res;
    }

    /**
     * @brief Get the number of threads in the pool.
     * @return size_t
     */
    size_t size() const noexcept { return threads_.size(); }

    /**
     * @brief Checks if thread affinity was successfully applied for all threads.
     * @return true if successful, false otherwise.
     */
    bool check_affinity_success() const noexcept { return affinity_success_; }

private:
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;

    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_;
    std::atomic<bool> affinity_success_;

    /**
     * @brief The main loop executed by each worker thread.
     */
    void worker_loop();

    /**
     * @brief Applies thread affinity to the current thread.
     * @param core_id The logical CPU core ID to bind to.
     * @return true if successful, false otherwise.
     */
    bool set_thread_affinity(int core_id);
};

} // namespace sonatadb::inference
