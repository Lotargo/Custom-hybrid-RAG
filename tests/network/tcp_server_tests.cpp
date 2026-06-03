#include <gtest/gtest.h>
#include "sonatadb/network/tcp_server.h"
#include "sonatadb/network/protocol.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

#include <thread>
#include <chrono>
#include <vector>

using namespace sonatadb::network;

TEST(TcpServerTests, PingPong) {
    int port = 9091;
    TcpServer server(port, 128);

    // Регистрируем обработчик Ping (возвращает ту же payload)
    server.register_handler(CommandID::Ping, [](const PacketHeader& header, std::span<const uint8_t> payload) {
        (void)header;
        return Protocol::serialize(CommandID::Ping, payload);
    });

    server.start();

    // Ждем, пока сервер запустится
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
    SOCKET client_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    ASSERT_NE(client_fd, INVALID_SOCKET);
#else
    int client_fd = socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_GE(client_fd, 0);
#endif

    struct sockaddr_in serv_addr;
    std::memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(static_cast<uint16_t>(port));

#ifdef _WIN32
    inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);
#else
    inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);
#endif

    ASSERT_EQ(connect(client_fd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)), 0);

    // Готовим пакет
    std::vector<uint8_t> payload = {1, 2, 3, 4};
    auto packet = Protocol::serialize(CommandID::Ping, payload);

    // Отправляем
#ifdef _WIN32
    int sent = send(client_fd, reinterpret_cast<const char*>(packet.data()), static_cast<int>(packet.size()), 0);
    ASSERT_EQ(sent, static_cast<int>(packet.size()));
#else
    ssize_t sent = send(client_fd, packet.data(), packet.size(), 0);
    ASSERT_EQ(sent, static_cast<ssize_t>(packet.size()));
#endif

    // Читаем ответ
    std::vector<uint8_t> response(1024);
#ifdef _WIN32
    int received = recv(client_fd, reinterpret_cast<char*>(response.data()), static_cast<int>(response.size()), 0);
    ASSERT_GT(received, 0);
#else
    ssize_t received = recv(client_fd, response.data(), response.size(), 0);
    ASSERT_GT(received, 0);
#endif

    // Проверяем
    auto parse_result = Protocol::parse_header(std::span<const uint8_t>(response.data(), static_cast<size_t>(received)));
    ASSERT_EQ(parse_result.status, Protocol::ParseStatus::Ok);
    auto header = parse_result.header.value();
    EXPECT_EQ(header.command, CommandID::Ping);
    EXPECT_EQ(header.length, payload.size());

#ifdef _WIN32
    closesocket(client_fd);
    WSACleanup();
#else
    close(client_fd);
#endif

    server.stop();
}
