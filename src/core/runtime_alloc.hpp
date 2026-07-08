#pragma once

#include "runtime_state.hpp"

#include <new>

namespace astral::core {

template<typename T, typename... Args>
inline T* runtime_new(Args&&... args) noexcept {
    void* mem = runtime_alloc(sizeof(T), alignof(T));
    if (mem == nullptr) {
        return nullptr;
    }
    return new (mem) T(static_cast<Args&&>(args)...);
}

template<typename T>
inline void runtime_delete(T* ptr) noexcept {
    if (ptr == nullptr) {
        return;
    }
    ptr->~T();
    runtime_free(ptr, sizeof(T), alignof(T));
}

template<typename T>
inline T* runtime_alloc_array(uint32_t count) noexcept {
    if (count == 0) {
        return nullptr;
    }
    const size_t bytes = static_cast<size_t>(count) * sizeof(T);
    void* mem = runtime_alloc(bytes, alignof(T));
    if (mem == nullptr) {
        return nullptr;
    }
    return static_cast<T*>(mem);
}

template<typename T>
inline void runtime_free_array(T* ptr, uint32_t count) noexcept {
    if (ptr == nullptr) {
        return;
    }
    const size_t bytes = static_cast<size_t>(count) * sizeof(T);
    runtime_free(ptr, bytes, alignof(T));
}

template <typename T> inline T* runtime_new_array(uint32_t count) noexcept {
  T* ptr = runtime_alloc_array<T>(count);
  if (ptr == nullptr) {
    return nullptr;
  }
  for (uint32_t i = 0; i < count; ++i) {
    new (ptr + i) T();
  }
  return ptr;
}

template <typename T> inline void runtime_delete_array(T* ptr, uint32_t count) noexcept {
  if (ptr == nullptr) {
    return;
  }
  for (uint32_t i = 0; i < count; ++i) {
    ptr[i].~T();
  }
  runtime_free_array(ptr, count);
}

} // namespace astral::core
