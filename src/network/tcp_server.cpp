#include "sonatadb/network/tcp_server.h"

#include <iostream>
#include <stdexcept>
#include <cstring>
#include <vector>

#ifdef SONATADB_USE_IO_URING
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/eventfd.h>
#else
// Windows or POSIX fallback
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/select.h>
#endif
#endif

namespace sonatadb::network {

#ifdef SONATADB_USE_IO_URING
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
#endif

TcpServer::TcpServer(int port, size_t ring_entries)
    : port_(port), ring_entries_(ring_entries) {
    
#ifdef SONATADB_USE_IO_URING
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        throw std::runtime_error("Failed to create socket");
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
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

    if (io_uring_queue_init(static_cast<unsigned>(ring_entries_), &ring_, 0) < 0) {
        close(server_fd_);
        throw std::runtime_error("Failed to initialize io_uring");
    }

    event_fd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (event_fd_ < 0) {
        io_uring_queue_exit(&ring_);
        close(server_fd_);
        throw std::runtime_error("Failed to create eventfd");
    }
#else
    // Windows or non-uring fallback
#ifdef _WIN32
    WSADATA wsaData;
    int wsa_res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (wsa_res != 0) {
        throw std::runtime_error("WSAStartup failed: " + std::to_string(wsa_res));
    }

    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) {
        WSACleanup();
        throw std::runtime_error("Failed to create socket: " + std::to_string(WSAGetLastError()));
    }
    server_fd_ = s;

    char opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#else
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        throw std::runtime_error("Failed to create socket");
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port_));

#ifdef _WIN32
    if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        closesocket(server_fd_);
        WSACleanup();
        throw std::runtime_error("Failed to bind socket: " + std::to_string(WSAGetLastError()));
    }

    if (listen(server_fd_, SOMAXCONN) == SOCKET_ERROR) {
        closesocket(server_fd_);
        WSACleanup();
        throw std::runtime_error("Failed to listen: " + std::to_string(WSAGetLastError()));
    }
#else
    if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(server_fd_);
        throw std::runtime_error("Failed to bind socket");
    }

    if (listen(server_fd_, SOMAXCONN) < 0) {
        close(server_fd_);
        throw std::runtime_error("Failed to listen");
    }
#endif
#endif
}

TcpServer::~TcpServer() {
    stop();
#ifdef SONATADB_USE_IO_URING
    io_uring_queue_exit(&ring_);
    close(server_fd_);
    close(event_fd_);
#else
#ifdef _WIN32
    if (server_fd_ != INVALID_SOCKET) {
        closesocket(server_fd_);
    }
    WSACleanup();
#else
    if (server_fd_ >= 0) {
        close(server_fd_);
    }
#endif
#endif
}

void TcpServer::register_handler(CommandID command, CommandHandler handler) {
    handlers_[command] = handler;
}

void TcpServer::start() {
    if (running_) return;
    running_ = true;
    worker_ = std::thread(&TcpServer::loop, this);
}

void TcpServer::stop() {
    if (!running_) return;
    running_ = false;

#ifdef SONATADB_USE_IO_URING
    // Пишем в eventfd, чтобы прервать io_uring_wait_cqe
    uint64_t val = 1;
    ssize_t written = write(event_fd_, &val, sizeof(val));
    (void)written;
#endif

    if (worker_.joinable()) {
        worker_.join();
    }
}

#ifdef SONATADB_USE_IO_URING
void TcpServer::add_accept() {
    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (!sqe) return;

    RequestData* req = new RequestData{EventType::Accept, nullptr};
    if (!client_addr_ptr_) {
        client_addr_ptr_ = ::malloc(sizeof(struct sockaddr_in));
    }
    client_addr_len_ = sizeof(struct sockaddr_in);
    io_uring_prep_accept(sqe, server_fd_, (struct sockaddr*)client_addr_ptr_, &client_addr_len_, 0);
    io_uring_sqe_set_data(sqe, req);
}

void TcpServer::add_read(ClientSession* session) {
    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (!sqe) return;

    auto buf = session->rx_buffer.write_buffer();
    if (buf.empty()) {
        std::cerr << "RX Buffer full for fd: " << session->fd << "\n";
        return;
    }

    RequestData* req = new RequestData{EventType::Read, session};
    io_uring_prep_recv(sqe, session->fd, buf.data(), buf.size(), 0);
    io_uring_sqe_set_data(sqe, req);
}

void TcpServer::add_write(ClientSession* session, std::span<const uint8_t> data) {
    if (data.empty()) return;

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

void TcpServer::handle_read(ClientSession* session, int res) {
    if (res <= 0) {
        close(session->fd);
        sessions_.erase(session->fd);
        return;
    }

    session->rx_buffer.commit_write(static_cast<size_t>(res));
    process_messages(session);

    if (sessions_.count(session->fd)) {
        add_read(session);
    }
}

void TcpServer::handle_write(ClientSession* session, int res) {
    if (res < 0) {
        std::cerr << "Write error on fd: " << session->fd << " res: " << res << "\n";
        close(session->fd);
        sessions_.erase(session->fd);
        return;
    }

    session->tx_buffer.advance(static_cast<size_t>(res));

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
#else
void TcpServer::add_write(ClientSession* session, std::span<const uint8_t> data) {
    if (data.empty()) return;
    
    size_t sent = 0;
    while (sent < data.size() && running_) {
#ifdef _WIN32
        int res = send(session->fd, reinterpret_cast<const char*>(data.data() + sent), static_cast<int>(data.size() - sent), 0);
#else
        ssize_t res = send(session->fd, data.data() + sent, data.size() - sent, 0);
#endif
        if (res < 0) {
#ifdef _WIN32
            int err = WSAGetLastError();
            if (err == WSAEWOULDBLOCK) {
                std::this_thread::yield();
                continue;
            }
#else
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::yield();
                continue;
            }
#endif
            break;
        }
        sent += static_cast<size_t>(res);
    }
}
#endif

void TcpServer::process_messages(ClientSession* session) {
    while (true) {
        if (session->rx_buffer.available() < HEADER_SIZE) {
            break;
        }

        uint8_t header_buf[HEADER_SIZE];
        session->rx_buffer.peek(std::span<uint8_t>(header_buf, HEADER_SIZE));

        auto parse_result = Protocol::parse_header(std::span<const uint8_t>(header_buf, HEADER_SIZE));

        if (parse_result.status == Protocol::ParseStatus::NeedMoreData) {
            break;
        } else if (parse_result.status == Protocol::ParseStatus::InvalidMagic) {
            std::cerr << "Invalid magic byte, closing connection\n";
#ifdef SONATADB_USE_IO_URING
            close(session->fd);
#else
#ifdef _WIN32
            closesocket(session->fd);
#else
            close(session->fd);
#endif
#endif
            sessions_.erase(session->fd);
            return;
        }

        const auto& header = parse_result.header.value();
        size_t total_packet_size = HEADER_SIZE + header.length;

        if (session->rx_buffer.available() < total_packet_size) {
            break;
        }

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

        session->rx_buffer.advance(total_packet_size);
    }
}

void TcpServer::loop() {
#ifdef SONATADB_USE_IO_URING
    struct io_uring_sqe* ev_sqe = io_uring_get_sqe(&ring_);
    if (ev_sqe) {
        RequestData* ev_req = new RequestData{EventType::StopSignal, nullptr};
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
            if (ret == -EINTR) continue;
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
                auto session = std::make_unique<ClientSession>(client_fd);
                ClientSession* ptr = session.get();
                sessions_[client_fd] = std::move(session);
                add_read(ptr);
            }
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
            delete req;
            io_uring_cqe_seen(&ring_, cqe);
            break;
        }

        delete req;
        io_uring_cqe_seen(&ring_, cqe);
        io_uring_submit(&ring_);
    }

    while (true) {
        int ret = io_uring_peek_cqe(&ring_, &cqe);
        if (ret < 0) break;
        RequestData* r = static_cast<RequestData*>(io_uring_cqe_get_data(cqe));
        if (r) delete r;
        io_uring_cqe_seen(&ring_, cqe);
    }

    if (client_addr_ptr_) {
        ::free(client_addr_ptr_);
        client_addr_ptr_ = nullptr;
    }
#else
    // Windows/POSIX select fallback loop
#ifdef _WIN32
    u_long mode = 1;
    ioctlsocket(server_fd_, FIONBIO, &mode);
#else
    int flags = fcntl(server_fd_, F_GETFL, 0);
    fcntl(server_fd_, F_SETFL, flags | O_NONBLOCK);
#endif

    while (running_) {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(server_fd_, &read_fds);

        auto max_fd = server_fd_;
        for (const auto& [fd, session] : sessions_) {
            FD_SET(fd, &read_fds);
            if (fd > max_fd) {
                max_fd = fd;
            }
        }

        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 20000; // 20ms check interval

#ifdef _WIN32
        int activity = select(0, &read_fds, nullptr, nullptr, &tv);
#else
        int activity = select(static_cast<int>(max_fd + 1), &read_fds, nullptr, nullptr, &tv);
#endif

        if (activity < 0) {
#ifdef _WIN32
            int err = WSAGetLastError();
            if (err == WSAEINTR) continue;
#else
            if (errno == EINTR) continue;
#endif
            break;
        }

        if (activity == 0) {
            continue;
        }

        if (FD_ISSET(server_fd_, &read_fds)) {
            struct sockaddr_in client_addr;
#ifdef _WIN32
            int addr_len = sizeof(client_addr);
            SOCKET client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &addr_len);
            if (client_fd != INVALID_SOCKET) {
                u_long c_mode = 1;
                ioctlsocket(client_fd, FIONBIO, &c_mode);
                sessions_[client_fd] = std::make_unique<ClientSession>(client_fd);
            }
#else
            socklen_t addr_len = sizeof(client_addr);
            int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &addr_len);
            if (client_fd >= 0) {
                int c_flags = fcntl(client_fd, F_GETFL, 0);
                fcntl(client_fd, F_SETFL, c_flags | O_NONBLOCK);
                sessions_[client_fd] = std::make_unique<ClientSession>(client_fd);
            }
#endif
        }

        std::vector<decltype(max_fd)> to_remove;
        for (const auto& [fd, session] : sessions_) {
            if (FD_ISSET(fd, &read_fds)) {
                auto buf = session->rx_buffer.write_buffer();
                if (buf.empty()) {
                    continue;
                }

#ifdef _WIN32
                int bytes_recv = recv(fd, reinterpret_cast<char*>(buf.data()), static_cast<int>(buf.size()), 0);
#else
                ssize_t bytes_recv = recv(fd, buf.data(), buf.size(), 0);
#endif

                if (bytes_recv > 0) {
                    session->rx_buffer.commit_write(static_cast<size_t>(bytes_recv));
                    process_messages(session.get());
                    // If session was erased inside process_messages, it will no longer exist in sessions_
                    if (sessions_.find(fd) == sessions_.end()) {
                        continue;
                    }
                } else {
                    to_remove.push_back(fd);
                }
            }
        }

        for (auto fd : to_remove) {
            if (sessions_.count(fd)) {
#ifdef _WIN32
                closesocket(fd);
#else
                close(fd);
#endif
                sessions_.erase(fd);
            }
        }
    }

    for (const auto& [fd, session] : sessions_) {
#ifdef _WIN32
        closesocket(fd);
#else
        close(fd);
#endif
    }
    sessions_.clear();
#endif
}

} // namespace sonatadb::network
