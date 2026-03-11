#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace sonatadb::network {

// Простой неблокирующий Ring Buffer для Zero-Allocation сетевого транспорта
// Память выделяется один раз при создании
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity);

    // Добавляет данные в буфер (копирование). Возвращает false, если места не хватает
    bool write(std::span<const uint8_t> data);

    // Возвращает span на доступные для чтения данные
    // Внимание: может вернуть лишь часть доступных данных, если они пересекают границу конца вектора.
    // Чтобы прочитать остаток, нужно будет вызвать read() еще раз после advance()
    std::span<const uint8_t> read() const;

    // Копирует до bytes байт из буфера в out_data, не сдвигая указатель чтения (Peek)
    // Полезно, если данные пересекают границу (wrap-around) и нужны единым куском
    size_t peek(std::span<uint8_t> out_data) const;

    // Сдвигает указатель чтения (помечает байты как прочитанные)
    void advance(size_t bytes);

    // Возвращает количество байт, доступных для чтения
    size_t available() const;

    // Возвращает количество свободного места для записи
    size_t free_space() const;

    // Сбрасывает указатели (очищает буфер)
    void clear();

    // Предоставляет сырой буфер для записи (например, для recv() или io_uring)
    // Возвращает span на непрерывный кусок памяти для записи.
    std::span<uint8_t> write_buffer();

    // Подтверждает, что внешняя функция (например recv) записала bytes в write_buffer()
    void commit_write(size_t bytes);

private:
    std::vector<uint8_t> buffer_;
    size_t read_pos_ = 0;
    size_t write_pos_ = 0;
    size_t capacity_ = 0;
    bool is_full_ = false;
};

} // namespace sonatadb::network
