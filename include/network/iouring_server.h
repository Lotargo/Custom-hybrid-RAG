#pragma once

#include "network/ring_buffer.h"
#include "network/protocol.h"

#include <liburing.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <thread>
#include <functional>

namespace rag_engine::network {

// Обработчик команды
using CommandHandler = std::function<std::vector<uint8_t>(const PacketHeader& header, std::span<const uint8_t> payload)>;

struct ClientSession {
    int fd;
    RingBuffer rx_buffer; // Буфер для приема
    RingBuffer tx_buffer; // Буфер для отправки (на будущее)

    ClientSession(int fd_in, size_t buf_size = 4096)
        : fd(fd_in), rx_buffer(buf_size), tx_buffer(buf_size) {}
};

class IoUringServer {
public:
    IoUringServer(int port, size_t ring_entries = 1024);
    ~IoUringServer();

    // Запуск сервера в отдельном потоке
    void start();

    // Остановка сервера
    void stop();

    // Регистрация обработчиков команд
    void register_handler(CommandID command, CommandHandler handler);

private:
    void loop();
    void add_accept();
    void add_read(ClientSession* session);
    void add_write(ClientSession* session, std::span<const uint8_t> data);
    void handle_read(ClientSession* session, int res);
    void handle_write(ClientSession* session, int res);
    void process_messages(ClientSession* session);

    int port_;
    int server_fd_;
    struct io_uring ring_;
    std::atomic<bool> running_{false};
    std::thread worker_;

    // Роутер команд
    std::unordered_map<CommandID, CommandHandler> handlers_;

    // Управление сессиями (выделяем на куче для стабильного указателя)
    std::unordered_map<int, std::unique_ptr<ClientSession>> sessions_;

    // Для accept
    struct sockaddr_in client_addr_;
    socklen_t client_addr_len_;

    // EventFD для прерывания io_uring_wait_cqe
    int event_fd_;
};

} // namespace rag_engine::network
