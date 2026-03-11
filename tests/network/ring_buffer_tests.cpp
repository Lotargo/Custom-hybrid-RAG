#include <gtest/gtest.h>
#include "network/ring_buffer.h"
#include <vector>

using namespace rag_engine::network;

TEST(RingBufferTests, WriteAndRead) {
    RingBuffer rb(10);
    std::vector<uint8_t> data = {1, 2, 3, 4, 5};

    EXPECT_TRUE(rb.write(data));
    EXPECT_EQ(rb.available(), 5);
    EXPECT_EQ(rb.free_space(), 5);

    auto read_span = rb.read();
    EXPECT_EQ(read_span.size(), 5);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(read_span[i], data[i]);
    }

    rb.advance(5);
    EXPECT_EQ(rb.available(), 0);
    EXPECT_EQ(rb.free_space(), 10);
}

TEST(RingBufferTests, WrapAround) {
    RingBuffer rb(5);
    std::vector<uint8_t> data1 = {1, 2, 3};
    std::vector<uint8_t> data2 = {4, 5};

    rb.write(data1);
    rb.advance(2); // Прочитали {1, 2}, осталось {3}

    rb.write(data2); // Записали {4, 5}. Теперь в буфере {3, 4, 5}

    EXPECT_EQ(rb.available(), 3);
    EXPECT_EQ(rb.free_space(), 2);

    auto read_span1 = rb.read();
    EXPECT_EQ(read_span1.size(), 3); // Зависит от внутренней реализации, может быть 1 или 3.
                                     // У нас wrap around, но 3 было записано последним элементом
                                     // Поэтому read_pos_ = 2. capacity = 5.
                                     // write_pos = 0. Значит read() вернет с 2 до 5 (размер 3).
    EXPECT_EQ(read_span1[0], 3);
    EXPECT_EQ(read_span1[1], 4);
    EXPECT_EQ(read_span1[2], 5);
}

TEST(RingBufferTests, WriteBufferAndCommit) {
    RingBuffer rb(10);
    auto buf = rb.write_buffer();
    EXPECT_EQ(buf.size(), 10);

    buf[0] = 42;
    buf[1] = 43;

    rb.commit_write(2);
    EXPECT_EQ(rb.available(), 2);

    auto read_buf = rb.read();
    EXPECT_EQ(read_buf[0], 42);
    EXPECT_EQ(read_buf[1], 43);
}
