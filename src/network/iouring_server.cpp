#include "sonatadb/network/iouring_server.h"

#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdexcept>
#include <cstring>
#include <sys/eventfd.h>

namespace sonatadb::network {

enum class EventType {
    Accept,
    Read,
    Write,
    StopSignal
};

struct RequestData {
    EventType type;
    ClientSession* session; // nullptr для Accept и StopSignal
};

IoUringServer::IoUringServer(int port, size_t ring_entries) : port_(port) {
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        throw std::runtime_error("Failed to create socket");
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port_));

    if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(server_fd_);
        throw std::runtime_error("Failed to bind socket");
    }

    if (listen(server_fd_, SOMAXCONN) < 0) {
        close(server_fd_);
        throw std::runtime_error("Failed to listen");
    }

    if (io_uring_queue_init(static_cast<unsigned>(ring_entries), &ring_, 0) < 0) {
        close(server_fd_);
        throw std::runtime_error("Failed to initialize io_uring");
    }

    event_fd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (event_fd_ < 0) {
        io_uring_queue_exit(&ring_);
        close(server_fd_);
        throw std::runtime_error("Failed to create eventfd");
    }
}

IoUringServer::~IoUringServer() {
    stop();
    io_uring_queue_exit(&ring_);
    close(server_fd_);
    close(event_fd_);
}

void IoUringServer::register_handler(CommandID command, CommandHandler handler) {
    handlers_[command] = handler;
}

void IoUringServer::start() {
    if (running_) return;
    running_ = true;
    worker_ = std::thread(&IoUringServer::loop, this);
}

void IoUringServer::stop() {
    if (!running_) return;
    running_ = false;

    // Пишем в eventfd, чтобы прервать io_uring_wait_cqe
    uint64_t val = 1;
    write(event_fd_, &val, sizeof(val));

    if (worker_.joinable()) {
        worker_.join();
    }
}

void IoUringServer::add_accept() {
    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (!sqe) return;

    RequestData* req = new RequestData{EventType::Accept, nullptr}; // Утечка, если не удалить в цикле
    client_addr_len_ = sizeof(client_addr_);
    io_uring_prep_accept(sqe, server_fd_, (struct sockaddr*)&client_addr_, &client_addr_len_, 0);
    io_uring_sqe_set_data(sqe, req);
}

void IoUringServer::add_read(ClientSession* session) {
    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (!sqe) return;

    auto buf = session->rx_buffer.write_buffer();
    if (buf.empty()) {
        // Буфер полон, нужно освободить или закрыть
        std::cerr << "RX Buffer full for fd: " << session->fd << "\n";
        return;
    }

    RequestData* req = new RequestData{EventType::Read, session};
    io_uring_prep_recv(sqe, session->fd, buf.data(), buf.size(), 0);
    io_uring_sqe_set_data(sqe, req);
}

void IoUringServer::add_write(ClientSession* session, std::span<const uint8_t> data) {
    if (data.empty()) return;

    // В простом варианте мы пишем в tx_buffer, а затем отправляем из tx_buffer.
    // Если tx_buffer заполнен, пакет может быть отброшен (или нужно закрыть сессию).
    if (!session->tx_buffer.write(data)) {
        std::cerr << "TX Buffer full for fd: " << session->fd << "\n";
        return;
    }

    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (!sqe) return;

    auto buf = session->tx_buffer.read();
    RequestData* req = new RequestData{EventType::Write, session};
    io_uring_prep_send(sqe, session->fd, buf.data(), buf.size(), 0);
    io_uring_sqe_set_data(sqe, req);
}

void IoUringServer::process_messages(ClientSession* session) {
    while (true) {
        if (session->rx_buffer.available() < HEADER_SIZE) {
            break; // Ждем хотя бы заголовок
        }

        // Извлекаем заголовок целиком (с учетом возможного wrap-around), не сдвигая указатель
        uint8_t header_buf[HEADER_SIZE];
        session->rx_buffer.peek(std::span<uint8_t>(header_buf, HEADER_SIZE));

        auto parse_result = Protocol::parse_header(std::span<const uint8_t>(header_buf, HEADER_SIZE));

        if (parse_result.status == Protocol::ParseStatus::NeedMoreData) {
            break;
        } else if (parse_result.status == Protocol::ParseStatus::InvalidMagic) {
            // Защита от DoS: клиент прислал мусор.
            // Закрываем соединение, чтобы не парсить невалидные данные дальше.
            std::cerr << "Invalid magic byte, closing connection " << session->fd << "\n";
            close(session->fd);
            sessions_.erase(session->fd);
            return;
        }

        const auto& header = parse_result.header.value();
        size_t total_packet_size = HEADER_SIZE + header.length;

        if (session->rx_buffer.available() < total_packet_size) {
            break; // Ждем payload целиком
        }

        // Пакет есть целиком. Вычитываем его в линейный буфер (копирование).
        // В будущем для больших пакетов можно оптимизировать через iovec,
        // но для начала собираем в вектор.
        std::vector<uint8_t> packet_data(total_packet_size);
        session->rx_buffer.peek(std::span<uint8_t>(packet_data.data(), total_packet_size));

        std::span<const uint8_t> payload(packet_data.data() + HEADER_SIZE, header.length);

        if (Protocol::validate_payload(header, payload)) {
            if (handlers_.count(header.command)) {
                auto response = handlers_[header.command](header, payload);
                add_write(session, response);
            }
        } else {
            std::cerr << "Invalid CRC32 from fd " << session->fd << "\n";
        }

        // Пакет обработан, можно смело сдвигать указатель
        session->rx_buffer.advance(total_packet_size);
    }
}

void IoUringServer::handle_read(ClientSession* session, int res) {
    if (res <= 0) {
        // Ошибка или соединение закрыто
        close(session->fd);
        sessions_.erase(session->fd);
        return;
    }

    session->rx_buffer.commit_write(static_cast<size_t>(res));
    process_messages(session);

    // Если клиент еще жив, слушаем дальше
    if (sessions_.count(session->fd)) {
        add_read(session);
    }
}

void IoUringServer::handle_write(ClientSession* session, int res) {
    if (res < 0) {
        // Ошибка отправки (может быть -EPIPE)
        std::cerr << "Write error on fd: " << session->fd << " res: " << res << "\n";
        close(session->fd);
        sessions_.erase(session->fd);
        return;
    }

    // Сдвигаем tx_buffer на количество отправленных байт
    session->tx_buffer.advance(static_cast<size_t>(res));

    // Если в буфере еще остались данные (не все отправилось за один раз),
    // ставим еще один запрос на отправку
    if (session->tx_buffer.available() > 0 && sessions_.count(session->fd)) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if (sqe) {
            auto buf = session->tx_buffer.read();
            RequestData* req = new RequestData{EventType::Write, session};
            io_uring_prep_send(sqe, session->fd, buf.data(), buf.size(), 0);
            io_uring_sqe_set_data(sqe, req);
        }
    }
}

void IoUringServer::loop() {
    // Добавляем чтение из eventfd для сигнала остановки
    struct io_uring_sqe* ev_sqe = io_uring_get_sqe(&ring_);
    if (ev_sqe) {
        // Мы выделяем RequestData на стеке или куче? На куче, как и все.
        RequestData* ev_req = new RequestData{EventType::StopSignal, nullptr};
        // Буфер для чтения из event_fd
        static uint64_t ev_buf;
        io_uring_prep_read(ev_sqe, event_fd_, &ev_buf, sizeof(ev_buf), 0);
        io_uring_sqe_set_data(ev_sqe, ev_req);
    }

    add_accept();
    io_uring_submit(&ring_);

    struct io_uring_cqe* cqe;
    while (running_) {
        int ret = io_uring_wait_cqe(&ring_, &cqe);
        if (ret < 0) {
            if (ret == -EINTR) continue; // Прерывание, игнорируем
            break;
        }

        RequestData* req = static_cast<RequestData*>(io_uring_cqe_get_data(cqe));
        if (!req) {
            io_uring_cqe_seen(&ring_, cqe);
            continue;
        }

        if (req->type == EventType::Accept) {
            int client_fd = cqe->res;
            if (client_fd >= 0) {
                // Создаем сессию
                auto session = std::make_unique<ClientSession>(client_fd);
                ClientSession* ptr = session.get();
                sessions_[client_fd] = std::move(session);

                // Добавляем чтение
                add_read(ptr);
            }
            // Снова ставим Accept
            if (running_) {
                add_accept();
            }
        } else if (req->type == EventType::Read) {
            if (sessions_.count(req->session->fd)) {
                handle_read(req->session, cqe->res);
            }
        } else if (req->type == EventType::Write) {
            if (sessions_.count(req->session->fd)) {
                handle_write(req->session, cqe->res);
            }
        } else if (req->type == EventType::StopSignal) {
            // Пришел сигнал на остановку, цикл прервется, так как running_ = false
            delete req;
            io_uring_cqe_seen(&ring_, cqe);
            break;
        }

        delete req; // Освобождаем память запроса
        io_uring_cqe_seen(&ring_, cqe);
        io_uring_submit(&ring_);
    }

    // Очистка оставшихся SQE/CQE, чтобы не было утечек RequestData
    while (true) {
        int ret = io_uring_peek_cqe(&ring_, &cqe);
        if (ret < 0) break;
        RequestData* r = static_cast<RequestData*>(io_uring_cqe_get_data(cqe));
        if (r) delete r;
        io_uring_cqe_seen(&ring_, cqe);
    }
}

} // namespace sonatadb::network
