#ifndef SONATADB_C_API_H
#define SONATADB_C_API_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
  #ifdef sonatadb_c_EXPORTS
    #define SONATADB_API __declspec(dllexport)
  #else
    #define SONATADB_API __declspec(dllimport)
  #endif
#else
  #define SONATADB_API
#endif

// ============================================================================
// Types
// ============================================================================

typedef struct sonatadb_hnsw_s sonatadb_hnsw_t;

typedef struct {
    uint32_t id;
    float distance;
} sonatadb_search_result_t;

// ============================================================================
// Error Handling
// ============================================================================

/**
 * @brief Retrieves the last error message for the calling thread.
 *
 * @param buffer The buffer to write the error message to.
 * @param buffer_size The size of the buffer.
 * @return The number of bytes written, excluding the null terminator.
 *         If 0 is returned, no error occurred or the buffer is too small.
 */
SONATADB_API size_t sonatadb_get_last_error_message(char* buffer, size_t buffer_size);

// ============================================================================
// HNSW Index
// ============================================================================

/**
 * @brief Creates a new HNSW index.
 *
 * @param filepath Path to the mmap file for the index.
 * @param dim Dimensionality of the vectors.
 * @param max_elements Maximum number of elements the index can hold.
 * @param M Number of bi-directional links created for every new element during element insertion.
 * @param ef_construction The size of the dynamic list for the nearest neighbors (used during the building).
 * @param read_only Whether to open the index in read-only mode.
 * @param index_out Pointer to the created index.
 * @return 0 on success, non-zero on error.
 */
SONATADB_API int sonatadb_hnsw_create(const char* filepath, size_t dim, size_t max_elements, size_t M, size_t ef_construction, bool read_only, sonatadb_hnsw_t** index_out);

/**
 * @brief Frees the given HNSW index.
 *
 * @param index The index to free.
 */
SONATADB_API void sonatadb_hnsw_free(sonatadb_hnsw_t* index);

/**
 * @brief Inserts a vector into the HNSW index.
 *
 * @param index The index to insert into.
 * @param vector Pointer to the vector data (array of floats).
 * @param id The ID to associate with the vector.
 * @return 0 on success, non-zero on error.
 */
SONATADB_API int sonatadb_hnsw_insert(sonatadb_hnsw_t* index, const float* vector, uint32_t id);

/**
 * @brief Searches the HNSW index for the nearest neighbors of a query vector.
 *
 * @param index The index to search in.
 * @param query_vector Pointer to the query vector data (array of floats).
 * @param k The number of nearest neighbors to return.
 * @param ef_search The size of the dynamic list for the nearest neighbors (used during search).
 * @param results Pointer to an array of sonatadb_search_result_t structs to store the results.
 *                Must be pre-allocated by the caller to hold at least `k` elements.
 * @param num_results_out Pointer to a variable to store the actual number of results returned.
 * @return 0 on success, non-zero on error.
 */
SONATADB_API int sonatadb_hnsw_search(sonatadb_hnsw_t* index, const float* query_vector, size_t k, size_t ef_search, sonatadb_search_result_t* results, size_t* num_results_out);

#ifdef __cplusplus
}
#endif

#endif // SONATADB_C_API_H