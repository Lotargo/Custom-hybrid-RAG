import os
import sys

# Add the bindings directory to sys.path so we can import sonatadb
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from bindings.python.sonatadb import HnswIndex, MetricType

def test_hnsw():
    print("Testing Python bindings for SonataDB C-API...")

    # We need a file path for mmap. We can use a temporary file.
    import tempfile

    with tempfile.TemporaryDirectory() as tempdir:
        filepath = os.path.join(tempdir, "test_index.bin")

        # 1. Create index
        print("Creating index...")
        index = HnswIndex(filepath, dim=4, max_elements=100)

        # 2. Insert vectors
        print("Inserting vectors...")
        vectors = [
            ([1.0, 0.0, 0.0, 0.0], 1),
            ([0.0, 1.0, 0.0, 0.0], 2),
            ([0.0, 0.0, 1.0, 0.0], 3),
            ([0.0, 0.0, 0.0, 1.0], 4),
            ([0.5, 0.5, 0.0, 0.0], 5),
        ]

        for vec, id in vectors:
            index.insert(vec, id)

        # 3. Search
        print("Searching...")
        query = [1.0, 0.0, 0.0, 0.0]
        results = index.search(query, k=2)

        print(f"Results for query {query}:")
        for i, (id, distance) in enumerate(results):
            print(f"  {i+1}. ID: {id}, Distance: {distance}")

        assert len(results) > 0, "No results returned"
        assert results[0][0] == 1, "Expected ID 1 to be the closest match"

        print("Python FFI tests passed successfully!")

if __name__ == "__main__":
    try:
        test_hnsw()
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)