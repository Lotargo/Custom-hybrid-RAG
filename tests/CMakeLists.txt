find_package(GTest CONFIG REQUIRED)

add_executable(rag_engine_tests
    test_main.cpp
    math/metrics_test.cpp
    math/metrics_avx2_test.cpp
)

target_link_libraries(rag_engine_tests PRIVATE
    GTest::gtest_main
    rag_engine_core
)

include(GoogleTest)
gtest_discover_tests(rag_engine_tests)
