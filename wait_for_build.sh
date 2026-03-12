#!/bin/bash
while ps aux | grep -v grep | grep -q cmake; do
    echo "Waiting for cmake..."
    sleep 10
done
echo "CMake finished. Running ninja tests..."
cd /app/build && ninja sonatadb_tests && ./tests/sonatadb_tests --gtest_filter=OnnxRunnerTest.*:InferencePoolTest.*
