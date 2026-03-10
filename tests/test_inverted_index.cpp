#include <gtest/gtest.h>
#include "rag_engine/index/inverted_index_builder.hpp"
#include "rag_engine/index/disk_inverted_index.hpp"
#include <filesystem>
#include <vector>

TEST(InvertedIndexTest, BuildAndSerialize) {
    using namespace rag_engine::index;

    std::string temp_file = "test_index.bin";

    {
        InvertedIndexBuilder builder("english");
        builder.add_document(1, "The quick brown fox");
        builder.add_document(2, "Jumps over the lazy dog");
        builder.add_document(3, "The quick dog");
        builder.add_document(4, "Foxes and dogs are animals");

        builder.serialize(temp_file);
    }

    {
        DiskInvertedIndex disk_index(temp_file);

        // "quick" -> doc 1 (tf 1), doc 3 (tf 1)
        auto postings_quick = disk_index.get_postings("quick");
        EXPECT_EQ(postings_quick.size(), 2);
        EXPECT_EQ(postings_quick[0].doc_id, 1);
        EXPECT_EQ(postings_quick[0].tf, 1);
        EXPECT_EQ(postings_quick[1].doc_id, 3);
        EXPECT_EQ(postings_quick[1].tf, 1);

        // "fox" -> doc 1 (tf 1), doc 4 ("foxes" stemmed to "fox", tf 1)
        auto postings_fox = disk_index.get_postings("fox");
        EXPECT_EQ(postings_fox.size(), 2);
        EXPECT_EQ(postings_fox[0].doc_id, 1);
        EXPECT_EQ(postings_fox[1].doc_id, 4);

        // "dog" -> doc 2, 3, 4
        auto postings_dog = disk_index.get_postings("dog");
        EXPECT_EQ(postings_dog.size(), 3);
        EXPECT_EQ(postings_dog[0].doc_id, 2);
        EXPECT_EQ(postings_dog[1].doc_id, 3);
        EXPECT_EQ(postings_dog[2].doc_id, 4);
    }

    std::filesystem::remove(temp_file);
}
