#include "rag_engine/index/document_filter.hpp"

namespace rag_engine::index {

DocumentFilter::DocumentFilter(const std::vector<uint32_t>& ids) {
    if (!ids.empty()) {
        bitmap_.addMany(ids.size(), ids.data());
    }
}

DocumentFilter::DocumentFilter(std::initializer_list<uint32_t> ids) {
    if (ids.size() > 0) {
        bitmap_.addMany(ids.size(), std::data(ids));
    }
}

DocumentFilter::DocumentFilter(roaring::Roaring bitmap) : bitmap_(std::move(bitmap)) {}

void DocumentFilter::add(uint32_t id) {
    bitmap_.add(id);
}

void DocumentFilter::remove(uint32_t id) {
    bitmap_.remove(id);
}

bool DocumentFilter::contains(uint32_t id) const {
    return bitmap_.contains(id);
}

uint64_t DocumentFilter::size() const {
    return bitmap_.cardinality();
}

DocumentFilter& DocumentFilter::operator&=(const DocumentFilter& other) {
    bitmap_ &= other.bitmap_;
    return *this;
}

DocumentFilter& DocumentFilter::operator|=(const DocumentFilter& other) {
    bitmap_ |= other.bitmap_;
    return *this;
}

DocumentFilter& DocumentFilter::operator-=(const DocumentFilter& other) {
    bitmap_ -= other.bitmap_;
    return *this;
}

DocumentFilter DocumentFilter::operator&(const DocumentFilter& other) const {
    return DocumentFilter(bitmap_ & other.bitmap_);
}

DocumentFilter DocumentFilter::operator|(const DocumentFilter& other) const {
    return DocumentFilter(bitmap_ | other.bitmap_);
}

DocumentFilter DocumentFilter::operator-(const DocumentFilter& other) const {
    return DocumentFilter(bitmap_ - other.bitmap_);
}

} // namespace rag_engine::index
