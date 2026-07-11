/**
 * string_builder.hpp - Stack-backed UTF-8 string builder
 */

#pragma once

#include "../core/runtime_state.hpp"
#include "../platform/compiler.hpp"
#include "utf8.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace astral::utf8 {

enum class StringOverflowPolicy : uint8_t {
  Spill,
  Truncate,
};

struct RuntimeStringAllocator {
  ASTRAL_FORCE_INLINE static void* allocate(size_t size, size_t align) noexcept {
    return core::runtime_alloc(size, align);
  }

  ASTRAL_FORCE_INLINE static void deallocate(void* ptr, size_t size, size_t align) noexcept {
    core::runtime_free(ptr, size, align);
  }
};

/**
 * Append-only builder with object-local storage and an optional cold spill.
 *
 * Each instance owns its storage and requires no synchronization. Appends that
 * fit in InlineCapacity do not allocate. Overflow either grows through the
 * allocator policy or truncates deterministically, depending on policy.
 */
template <uint32_t InlineCapacity = 256, typename SpillAllocator = RuntimeStringAllocator,
          StringOverflowPolicy OverflowPolicy = StringOverflowPolicy::Spill>
class StringBuilder {
  static_assert(InlineCapacity > 0, "StringBuilder requires inline storage");
  static_assert(InlineCapacity < (std::numeric_limits<uint32_t>::max)(),
                "StringBuilder inline capacity is too large");

public:
  StringBuilder() noexcept
      : buffer_(inline_buffer_), capacity_(InlineCapacity), length_(0), spilled_(false),
        truncated_(false) {
    inline_buffer_[0] = '\0';
  }

  ~StringBuilder() noexcept {
    if constexpr (OverflowPolicy == StringOverflowPolicy::Spill) {
      release_spill();
    }
  }

  StringBuilder(const StringBuilder&) = delete;
  StringBuilder& operator=(const StringBuilder&) = delete;
  StringBuilder(StringBuilder&&) = delete;
  StringBuilder& operator=(StringBuilder&&) = delete;

  bool append(Span utf8) noexcept {
    if (utf8.len == 0) {
      return true;
    }

    if (utf8.len <= capacity_ - length_)
      ASTRAL_LIKELY {
        copy_append(utf8.data, utf8.len);
        return true;
      }

    if constexpr (OverflowPolicy == StringOverflowPolicy::Spill) {
      if (grow_for(utf8.len)) {
        copy_append(utf8.data, utf8.len);
        return true;
      }
    }

    const uint32_t available = capacity_ - length_;
    const uint32_t copy_bytes = utf8.len < available ? utf8.len : available;
    for (uint32_t i = 0; i < copy_bytes; ++i) {
      buffer_[length_ + i] = utf8.data[i];
    }
    length_ += copy_bytes;
    buffer_[length_] = '\0';
    truncated_ = true;
    return false;
  }

  template <size_t Size> bool append_literal(const char (&text)[Size]) noexcept {
    static_assert(Size > 0, "string literal must include a terminator");
    static_assert(Size - 1u <= (std::numeric_limits<uint32_t>::max)(),
                  "string literal is too large");
    return append(Span(reinterpret_cast<const uint8_t*>(text), static_cast<uint32_t>(Size - 1u)));
  }

  bool append_char(char value) noexcept {
    const uint8_t byte = static_cast<uint8_t>(value);
    return append(Span(&byte, 1));
  }

  bool append_u32(uint32_t value) noexcept { return append_unsigned(value); }
  bool append_u64(uint64_t value) noexcept { return append_unsigned(value); }
  bool append_i32(int32_t value) noexcept { return append_signed(value); }
  bool append_i64(int64_t value) noexcept { return append_signed(value); }

  bool append_f32(float value, uint32_t decimals = 2) noexcept {
    constexpr uint32_t kMaxDecimals = std::numeric_limits<float>::max_digits10;
    if (decimals > kMaxDecimals) {
      return false;
    }

    char text[64];
    const auto result = std::to_chars(text, text + sizeof(text), value, std::chars_format::fixed,
                                      static_cast<int>(decimals));
    if (result.ec != std::errc{}) {
      return false;
    }

    const uint32_t size = static_cast<uint32_t>(result.ptr - text);
    return append(Span(reinterpret_cast<const uint8_t*>(text), size));
  }

  Span freeze() const noexcept { return Span(buffer_, length_); }
  const char* c_str() const noexcept { return reinterpret_cast<const char*>(buffer_); }
  const uint8_t* data() const noexcept { return buffer_; }

  void reset() noexcept {
    length_ = 0;
    truncated_ = false;
    buffer_[0] = '\0';
  }

  uint32_t length() const noexcept { return length_; }
  uint32_t capacity() const noexcept { return capacity_; }
  bool empty() const noexcept { return length_ == 0; }
  bool spilled() const noexcept { return spilled_; }
  bool truncated() const noexcept { return truncated_; }

private:
  template <typename UInt> bool append_unsigned(UInt value) noexcept {
    static_assert(std::is_unsigned<UInt>::value, "unsigned integer required");
    char text[std::numeric_limits<UInt>::digits10 + 1];
    char* cursor = text + sizeof(text);
    do {
      const UInt quotient = value / 10;
      *--cursor = static_cast<char>('0' + (value - quotient * 10));
      value = quotient;
    } while (value != 0);

    const uint32_t size = static_cast<uint32_t>(text + sizeof(text) - cursor);
    return append(Span(reinterpret_cast<const uint8_t*>(cursor), size));
  }

  template <typename Int> bool append_signed(Int value) noexcept {
    static_assert(std::is_signed<Int>::value, "signed integer required");
    using UInt = typename std::make_unsigned<Int>::type;

    char text[std::numeric_limits<UInt>::digits10 + 2];
    char* cursor = text + sizeof(text);
    UInt magnitude = static_cast<UInt>(value);
    if (value < 0) {
      magnitude = UInt{0} - magnitude;
    }

    do {
      const UInt quotient = magnitude / 10;
      *--cursor = static_cast<char>('0' + (magnitude - quotient * 10));
      magnitude = quotient;
    } while (magnitude != 0);

    if (value < 0) {
      *--cursor = '-';
    }

    const uint32_t size = static_cast<uint32_t>(text + sizeof(text) - cursor);
    return append(Span(reinterpret_cast<const uint8_t*>(cursor), size));
  }

  ASTRAL_FORCE_INLINE void copy_append(const uint8_t* src, uint32_t size) noexcept {
    std::memcpy(buffer_ + length_, src, size);
    length_ += size;
    buffer_[length_] = '\0';
  }

  bool grow_for(uint32_t additional) noexcept {
    const uint64_t required = static_cast<uint64_t>(length_) + additional;
    if (required > (std::numeric_limits<uint32_t>::max)()) {
      return false;
    }

    uint32_t new_capacity = capacity_;
    while (new_capacity < required) {
      if (new_capacity > (std::numeric_limits<uint32_t>::max)() / 2u) {
        new_capacity = static_cast<uint32_t>(required);
        break;
      }
      new_capacity *= 2u;
    }

    if (new_capacity >= (std::numeric_limits<size_t>::max)()) {
      return false;
    }
    const size_t allocation_size = static_cast<size_t>(new_capacity) + 1u;
    auto* next = static_cast<uint8_t*>(SpillAllocator::allocate(allocation_size, alignof(uint8_t)));
    if (next == nullptr) {
      return false;
    }

    std::memcpy(next, buffer_, static_cast<size_t>(length_) + 1u);
    if (spilled_) {
      SpillAllocator::deallocate(buffer_, static_cast<size_t>(capacity_) + 1u, alignof(uint8_t));
    }

    buffer_ = next;
    capacity_ = new_capacity;
    spilled_ = true;
    return true;
  }

  void release_spill() noexcept {
    if (!spilled_) {
      return;
    }
    SpillAllocator::deallocate(buffer_, static_cast<size_t>(capacity_) + 1u, alignof(uint8_t));
  }

  uint8_t inline_buffer_[static_cast<size_t>(InlineCapacity) + 1u];
  uint8_t* buffer_;
  uint32_t capacity_;
  uint32_t length_;
  bool spilled_;
  bool truncated_;
};

template <uint32_t InlineCapacity = 256, typename SpillAllocator = RuntimeStringAllocator>
using StackStringBuilder =
    StringBuilder<InlineCapacity, SpillAllocator, StringOverflowPolicy::Truncate>;

} // namespace astral::utf8
