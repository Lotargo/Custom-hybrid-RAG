#include <gtest/gtest.h>
#include "network/iouring_server.h"
#include "network/protocol.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <thread>
#include <chrono>

using namespace rag_engine::network;

TEST(IoUringServerTests, PingPong) {
    int port = 9091;
    IoUringServer server(port, 128);

    // Регистрируем обработчик Ping (возвращает ту же payload)
    server.register_handler(CommandID::Ping, [](const PacketHeader& header, std::span<const uint8_t> payload) {
        return Protocol::serialize(CommandID::Ping, payload);
    });

    server.start();

    // Ждем, пока сервер запустится
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Подключаемся клиентом (синхронно)
    int client_fd = socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_GE(client_fd, 0);

    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);

    ASSERT_EQ(connect(client_fd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)), 0);

    // Готовим пакет
    std::vector<uint8_t> payload = {1, 2, 3, 4};
    auto packet = Protocol::serialize(CommandID::Ping, payload);

    // Отправляем
    ssize_t sent = send(client_fd, packet.data(), packet.size(), 0);
    ASSERT_EQ(sent, packet.size());

    // Читаем ответ
    std::vector<uint8_t> response(1024);
    ssize_t received = recv(client_fd, response.data(), response.size(), 0);
    ASSERT_GT(received, 0);

    // Проверяем
    auto parse_result = Protocol::parse_header(std::span<const uint8_t>(response.data(), received));
    ASSERT_EQ(parse_result.status, Protocol::ParseStatus::Ok);
    auto header = parse_result.header.value();
    EXPECT_EQ(header.command, CommandID::Ping);
    EXPECT_EQ(header.length, payload.size());

    close(client_fd);
    server.stop();
}
