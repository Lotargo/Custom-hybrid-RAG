#include "sonatadb/c_api/sonatadb.h"
#include "sonatadb/index/hnsw.hpp"
#include <string>
#include <exception>
#include <cstring>
#include <vector>

// Thread-local buffer for the last error message
thread_local std::string tls_last_error_message;

// Helper to set the last error message
static void set_last_error(const std::string& msg) {
    tls_last_error_message = msg;
}

extern "C" {

size_t sonatadb_get_last_error_message(char* buffer, size_t buffer_size) {
    if (tls_last_error_message.empty() || buffer == nullptr || buffer_size == 0) {
        if (buffer && buffer_size > 0) {
            buffer[0] = '\0';
        }
        return 0;
    }

    size_t copy_len = std::min(tls_last_error_message.length(), buffer_size - 1);
    std::memcpy(buffer, tls_last_error_message.c_str(), copy_len);
    buffer[copy_len] = '\0';
    return copy_len;
}

int sonatadb_hnsw_create(const char* filepath, size_t dim, size_t max_elements, size_t M, size_t ef_construction, bool read_only, sonatadb_hnsw_t** index_out) {
    if (!filepath || !index_out) {
        set_last_error("Invalid arguments: filepath or index_out is null");
        return -1;
    }

    try {
        sonatadb::index::HnswConfig config;
        config.dim = dim;
        config.max_elements = max_elements;
        config.M = M;
        config.ef_construction = ef_construction;

        auto* index = new sonatadb::index::HnswIndex(filepath, config, read_only);
        *index_out = reinterpret_cast<sonatadb_hnsw_t*>(index);

        set_last_error("");
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return -1;
    } catch (...) {
        set_last_error("Unknown exception during sonatadb_hnsw_create");
        return -1;
    }
}

void sonatadb_hnsw_free(sonatadb_hnsw_t* index) {
    if (index) {
        auto* hnsw_idx = reinterpret_cast<sonatadb::index::HnswIndex*>(index);
        delete hnsw_idx;
    }
}

int sonatadb_hnsw_insert(sonatadb_hnsw_t* index, const float* vector, uint32_t id) {
    if (!index || !vector) {
        set_last_error("Invalid arguments: index or vector is null");
        return -1;
    }

    try {
        auto* hnsw_idx = reinterpret_cast<sonatadb::index::HnswIndex*>(index);
        // Assuming we need to know the dim here, unfortunately the C++ API requires a span.
        // We will need to peek into the index or just pass the vector if we know its size.
        // Wait, HnswIndex has `add(uint32_t id, std::span<const float> vector)`.
        // We can't easily get dim from HnswIndex public API. Let's add a `dim()` getter or assume it is known, or just use a hack since we know we need `dim`.
        // Actually, we can just use `sizeof` or similar? No, `span` needs the exact count of elements.
        // Let's modify `HnswIndex` to expose `dim()` if it doesn't, or we can just access it.
        // Oh wait, `HnswIndex` doesn't have a `dim()` getter right now.
        // Let's just create a span by knowing the config from a wrapper struct, or add a getter.
        // For now, let's look at `HnswIndex` private members. `config_.dim`.
        // It's better to add a `dim()` method to `HnswIndex`. Let's do that in a moment.
        // I will implement a quick workaround here to call a method that I will add to `hnsw.hpp`.
        std::span<const float> vec_span(vector, hnsw_idx->getDim());
        hnsw_idx->add(id, vec_span);

        set_last_error("");
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return -1;
    } catch (...) {
        set_last_error("Unknown exception during sonatadb_hnsw_insert");
        return -1;
    }
}

int sonatadb_hnsw_search(sonatadb_hnsw_t* index, const float* query_vector, size_t k, size_t ef_search, sonatadb_search_result_t* results, size_t* num_results_out) {
    if (!index || !query_vector || !results || !num_results_out) {
        set_last_error("Invalid arguments to sonatadb_hnsw_search");
        return -1;
    }

    try {
        auto* hnsw_idx = reinterpret_cast<sonatadb::index::HnswIndex*>(index);
        std::span<const float> query_span(query_vector, hnsw_idx->getDim());

        auto cpp_results = hnsw_idx->search(query_span, k, ef_search);

        size_t count = std::min(k, cpp_results.size());
        for (size_t i = 0; i < count; ++i) {
            results[i].id = cpp_results[i].id;
            results[i].distance = cpp_results[i].distance;
        }

        *num_results_out = count;
        set_last_error("");
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return -1;
    } catch (...) {
        set_last_error("Unknown exception during sonatadb_hnsw_search");
        return -1;
    }
}

} // extern "C"