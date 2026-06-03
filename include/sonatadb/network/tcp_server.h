#pragma once

#include "sonatadb/network/ring_buffer.h"
#include "sonatadb/network/protocol.h"

#include <unordered_map>
#include <memory>
#include <atomic>
#include <thread>
#include <functional>
#include <span>
#include <vector>

namespace sonatadb::network {

// Обработчик команды
using CommandHandler = std::function<std::vector<uint8_t>(const PacketHeader& header, std::span<const uint8_t> payload)>;

struct ClientSession {
#ifdef _WIN32
    uintptr_t fd; // SOCKET type in Winsock is uintptr_t
#else
    int fd;
#endif
    RingBuffer rx_buffer; // Буфер для приема
    RingBuffer tx_buffer; // Буфер для отправки

#ifdef _WIN32
    ClientSession(uintptr_t fd_in, size_t buf_size = 4096)
#else
    ClientSession(int fd_in, size_t buf_size = 4096)
#endif
        : fd(fd_in), rx_buffer(buf_size), tx_buffer(buf_size) {}
};

class TcpServer {
public:
    TcpServer(int port, size_t ring_entries = 1024);
    ~TcpServer();

    // Запуск сервера в отдельном потоке
    void start();

    // Остановка сервера
    void stop();

    // Регистрация обработчиков команд
    void register_handler(CommandID command, CommandHandler handler);

private:
    void loop();
    void process_messages(ClientSession* session);

#ifdef SONATADB_USE_IO_URING
    void add_accept();
    void add_read(ClientSession* session);
    void add_write(ClientSession* session, std::span<const uint8_t> data);
    void handle_read(ClientSession* session, int res);
    void handle_write(ClientSession* session, int res);

    int server_fd_;
    // Using void* to avoid putting liburing structs directly in the header if not needed,
    // but io_uring itself is lightweight. We can keep it or use void* pointers.
    // Let's use the actual type to match the previous header.
    struct io_uring ring_;
    int event_fd_;
    // For accept
    void* client_addr_ptr_ = nullptr;
    uint32_t client_addr_len_ = 0;
#else
    // Windows/Sockets fallback members
#ifdef _WIN32
    uintptr_t server_fd_; // SOCKET
    uintptr_t stop_pipe_read_; // For interrupting select
    uintptr_t stop_pipe_write_;
#else
    int server_fd_;
    int stop_pipe_read_;
    int stop_pipe_write_;
#endif
    void add_write(ClientSession* session, std::span<const uint8_t> data);
#endif

    int port_;
    size_t ring_entries_;
    std::atomic<bool> running_{false};
    std::thread worker_;

    // Роутер команд
    std::unordered_map<CommandID, CommandHandler> handlers_;

    // Управление сессиями (выделяем на куче для стабильного указателя)
#ifdef _WIN32
    std::unordered_map<uintptr_t, std::unique_ptr<ClientSession>> sessions_;
#else
    std::unordered_map<int, std::unique_ptr<ClientSession>> sessions_;
#endif
};

// Алиас для обратной совместимости
using IoUringServer = TcpServer;

} // namespace sonatadb::network
