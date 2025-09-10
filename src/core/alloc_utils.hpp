#pragma once

#include "../../include/astral_rt.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

#if defined(_WIN32)
  #include <malloc.h>
#endif

namespace astral::core {

inline void* alloc_raw(AstralAllocator a, size_t size, size_t align) noexcept {
    if (size == 0) {
        return nullptr;
    }
    if (a.alloc) {
        return a.alloc(a.user, size, align);
    }

    const size_t max_align = alignof(std::max_align_t);
    if (align <= max_align) {
        return std::malloc(size);
    }

#if defined(_WIN32)
    return _aligned_malloc(size, align);
#else
    void* p = nullptr;
    if (posix_memalign(&p, align, size) != 0) {
        return nullptr;
    }
    return p;
#endif
}

inline void free_raw(AstralAllocator a, void* ptr, size_t size, size_t align) noexcept {
    if (ptr == nullptr) {
        return;
    }
    if (a.free) {
        a.free(a.user, ptr, size, align);
        return;
    }
    (void)size;

#if defined(_WIN32)
    const size_t max_align = alignof(std::max_align_t);
    if (align > max_align) {
        _aligned_free(ptr);
        return;
    }
#endif
    (void)align;
    std::free(ptr);
}

template<typename T>
inline T* alloc_array(AstralAllocator a, uint32_t count) noexcept {
    if (count == 0) {
        return nullptr;
    }

    const size_t bytes = static_cast<size_t>(count) * sizeof(T);
    void* mem = alloc_raw(a, bytes, alignof(T));
    if (mem == nullptr) {
        return nullptr;
    }

    T* p = static_cast<T*>(mem);
    for (uint32_t i = 0; i < count; ++i) {
        new (p + i) T();
    }
    return p;
}

template<typename T>
inline void free_array(AstralAllocator a, T* ptr, uint32_t count) noexcept {
    if (ptr == nullptr) {
        return;
    }

    for (uint32_t i = 0; i < count; ++i) {
        ptr[i].~T();
    }

    const size_t bytes = static_cast<size_t>(count) * sizeof(T);
    free_raw(a, ptr, bytes, alignof(T));
}

template<typename T>
class ScopedArray {
public:
    ScopedArray() = default;
    ScopedArray(AstralAllocator a, uint32_t count) noexcept : alloc_(a), ptr_(alloc_array<T>(a, count)), count_(ptr_ ? count : 0u) {}

    ~ScopedArray() noexcept { reset(); }

    ScopedArray(const ScopedArray&) = delete;
    ScopedArray& operator=(const ScopedArray&) = delete;
    ScopedArray(ScopedArray&&) = delete;
    ScopedArray& operator=(ScopedArray&&) = delete;

    T* get() const noexcept { return ptr_; }
    uint32_t count() const noexcept { return count_; }

    T* release() noexcept {
        T* out = ptr_;
        ptr_ = nullptr;
        count_ = 0;
        return out;
    }

    void reset() noexcept {
        if (ptr_ != nullptr) {
            free_array(alloc_, ptr_, count_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

private:
    AstralAllocator alloc_{nullptr, nullptr, nullptr};
    T* ptr_{nullptr};
    uint32_t count_{0};
};

} // namespace astral::core
