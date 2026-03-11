import ctypes
import os
import sys
from typing import List, Tuple
from enum import IntEnum

class MetricType(IntEnum):
    L2 = 0
    COSINE = 1

class SearchResult(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("distance", ctypes.c_float)
    ]

class SonataDBError(Exception):
    pass

class SonataDBAPI:
    def __init__(self):
        # Determine library extension based on platform
        lib_name = "libsonatadb_c.so"
        if sys.platform == "darwin":
            lib_name = "libsonatadb_c.dylib"
        elif sys.platform == "win32":
            lib_name = "sonatadb_c.dll"

        # Try to load the library from common paths
        search_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "build", "src"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "build"),
            ".",
            "build/src",
            "build"
        ]

        self.lib = None
        for path in search_paths:
            full_path = os.path.abspath(os.path.join(path, lib_name))
            if os.path.exists(full_path):
                try:
                    self.lib = ctypes.CDLL(full_path)
                    break
                except OSError as e:
                    print(f"Found library at {full_path} but failed to load: {e}")
                    pass

        if self.lib is None:
            raise RuntimeError(f"Could not find {lib_name} in expected locations. Make sure to build the project first.")

        self._setup_argtypes()

    def _setup_argtypes(self):
        # sonatadb_get_last_error_message
        self.lib.sonatadb_get_last_error_message.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self.lib.sonatadb_get_last_error_message.restype = ctypes.c_size_t

        # sonatadb_hnsw_create
        self.lib.sonatadb_hnsw_create.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_bool, ctypes.POINTER(ctypes.c_void_p)
        ]
        self.lib.sonatadb_hnsw_create.restype = ctypes.c_int

        # sonatadb_hnsw_free
        self.lib.sonatadb_hnsw_free.argtypes = [ctypes.c_void_p]
        self.lib.sonatadb_hnsw_free.restype = None

        # sonatadb_hnsw_insert
        self.lib.sonatadb_hnsw_insert.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
        self.lib.sonatadb_hnsw_insert.restype = ctypes.c_int

        # sonatadb_hnsw_search
        self.lib.sonatadb_hnsw_search.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(SearchResult), ctypes.POINTER(ctypes.c_size_t)
        ]
        self.lib.sonatadb_hnsw_search.restype = ctypes.c_int

    def check_error(self, code: int):
        if code != 0:
            buffer = ctypes.create_string_buffer(1024)
            self.lib.sonatadb_get_last_error_message(buffer, 1024)
            error_msg = buffer.value.decode('utf-8')
            raise SonataDBError(error_msg)

api = SonataDBAPI()

class HnswIndex:
    def __init__(self, filepath: str, dim: int, max_elements: int, M: int = 16, ef_construction: int = 200, read_only: bool = False):
        self.dim = dim
        self._ptr = ctypes.c_void_p()

        path_bytes = filepath.encode('utf-8')
        res = api.lib.sonatadb_hnsw_create(
            path_bytes, dim, max_elements, M, ef_construction, read_only, ctypes.byref(self._ptr)
        )
        api.check_error(res)

    def __del__(self):
        if self._ptr:
            api.lib.sonatadb_hnsw_free(self._ptr)
            self._ptr = None

    def insert(self, vector: List[float], id: int):
        if len(vector) != self.dim:
            raise ValueError(f"Vector dimension must be {self.dim}")

        VectorArray = ctypes.c_float * self.dim
        c_vector = VectorArray(*vector)

        res = api.lib.sonatadb_hnsw_insert(self._ptr, ctypes.cast(c_vector, ctypes.POINTER(ctypes.c_float)), id)
        api.check_error(res)

    def search(self, query: List[float], k: int = 10, ef_search: int = 50) -> List[Tuple[int, float]]:
        if len(query) != self.dim:
            raise ValueError(f"Query dimension must be {self.dim}")

        VectorArray = ctypes.c_float * self.dim
        c_query = VectorArray(*query)

        ResultsArray = SearchResult * k
        c_results = ResultsArray()
        c_num_results = ctypes.c_size_t(0)

        res = api.lib.sonatadb_hnsw_search(
            self._ptr, ctypes.cast(c_query, ctypes.POINTER(ctypes.c_float)), k, ef_search,
            c_results, ctypes.byref(c_num_results)
        )
        api.check_error(res)

        return [(c_results[i].id, c_results[i].distance) for i in range(c_num_results.value)]