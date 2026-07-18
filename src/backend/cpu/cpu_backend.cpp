/**
 * cpu_backend.cpp - CPU Backend Implementation
 *
 * llama.cpp integration for CPU inference.
 *
 * Model lifetime:
 * - model_load(): loads GGUF weights into a CpuModel (shareable)
 * - session_create(): creates a CpuSession (llama_context) per Astral session
 *
 * Hot-path constraints:
 * - No allocations in session_feed()/session_accept()/session_logits()
 * - Avoid llama_batch_init/free (heap); use llama_batch_get_one() (no alloc)
 */

#include "cpu_backend.hpp"

#include "../../core/runtime_state.hpp"
#include "../../core/runtime_alloc.hpp"
#include "../../core/model_sources.hpp"
#include "../../core/model_load_config.hpp"
#include "../../utils/trace.hpp"
#include <llama.h>
#include <ggml-backend.h>
#if defined(ASTRAL_ENABLE_CUDA) && ASTRAL_ENABLE_CUDA
#include <ggml-cuda.h>
#endif
#if ASTRAL_ENABLE_MTMD
#include "mtmd.h"
#include "mtmd-helper.h"
#endif

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>
#include "json-schema-to-grammar.h"

#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(_MSC_VER)
#include <malloc.h>
#define ASTRAL_ALLOCA _alloca
#else
#include <alloca.h>
#define ASTRAL_ALLOCA alloca
#endif

namespace astral::backend {

static constexpr uint32_t kCpuStateMagic = 0x41535443u; // 'ASTC'
static constexpr uint16_t kCpuStateVersion = 1;

struct CpuStateHeaderV1 {
    uint32_t magic;
    uint16_t version;
    uint16_t header_bytes;
    uint64_t llama_bytes;
    uint32_t active_slot;
    uint32_t n_slots;
    int32_t slot_last_token[32];
    uint8_t slot_last_token_valid[32];
};

static_assert(sizeof(CpuStateHeaderV1) == 184, "CpuStateHeaderV1 size changed");
static_assert((sizeof(CpuStateHeaderV1) % 8) == 0, "CpuStateHeaderV1 must be 8-byte aligned");

namespace {

constexpr uint32_t kCpuMaxSlots = 32;

std::atomic<uint32_t> g_llama_backend_refs{0};
std::atomic<int> g_llama_log_threshold{GGML_LOG_LEVEL_WARN};

void llama_log_callback(ggml_log_level level, const char* text, void*) {
    const int threshold = g_llama_log_threshold.load(std::memory_order_relaxed);
    if (threshold < 0) {
        return;
    }
    if (static_cast<int>(level) >= threshold) {
        if (text != nullptr) {
            std::fputs(text, stderr);
        }
    }
}

#if defined(ASTRAL_ENABLE_CUDA) && ASTRAL_ENABLE_CUDA
static bool collect_cuda_devices(std::vector<ggml_backend_dev_t>& out) {
    ggml_backend_reg_t reg = ggml_backend_cuda_reg();
    if (!reg) {
        return false;
    }
    const size_t count = ggml_backend_reg_dev_count(reg);
    if (count == 0) {
        return false;
    }
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);
        if (dev != nullptr) {
            out.push_back(dev);
        }
    }
    return !out.empty();
}
#endif

static bool apply_gpu_config(const AstralModelDesc* desc,
                             llama_model_params& params,
                             std::vector<ggml_backend_dev_t>& out_devices,
                             std::vector<float>& out_tensor_split,
                             AstralErr* out_err) {
    if (desc == nullptr || desc->size != sizeof(AstralModelDesc)) {
        return true;
    }

    const uint32_t flags = desc->gpu_flags;
    if (flags == 0) {
        return true;
    }

#if !defined(ASTRAL_ENABLE_CUDA) || !ASTRAL_ENABLE_CUDA
    (void)params;
    (void)out_devices;
    (void)out_tensor_split;
    (void)out_err;
    return true;
#else
    if (flags & ASTRAL_GPU_CFG_SPLIT_MODE) {
        switch (desc->gpu_split_mode) {
            case ASTRAL_GPU_SPLIT_NONE:
                params.split_mode = LLAMA_SPLIT_MODE_NONE;
                break;
            case ASTRAL_GPU_SPLIT_LAYER:
                params.split_mode = LLAMA_SPLIT_MODE_LAYER;
                break;
            case ASTRAL_GPU_SPLIT_ROW:
                params.split_mode = LLAMA_SPLIT_MODE_ROW;
                break;
            default:
                if (out_err) {
                    *out_err = ASTRAL_E_INVALID;
                }
                return false;
        }
    }

    std::vector<ggml_backend_dev_t> cuda_devices;
    std::vector<int32_t> selected_indices;

    if (flags & (ASTRAL_GPU_CFG_DEVICES | ASTRAL_GPU_CFG_DEVICE_MASK)) {
        if (!collect_cuda_devices(cuda_devices)) {
            if (out_err) {
                *out_err = ASTRAL_E_BACKEND;
            }
            return false;
        }
    }

    if (flags & ASTRAL_GPU_CFG_DEVICES) {
        if (desc->gpu_devices == nullptr || desc->gpu_device_count == 0) {
            if (out_err) {
                *out_err = ASTRAL_E_INVALID;
            }
            return false;
        }
        out_devices.reserve(static_cast<size_t>(desc->gpu_device_count) + 1);
        selected_indices.reserve(static_cast<size_t>(desc->gpu_device_count));
        for (uint32_t i = 0; i < desc->gpu_device_count; ++i) {
            const int32_t idx = desc->gpu_devices[i];
            if (idx < 0 || static_cast<size_t>(idx) >= cuda_devices.size()) {
                if (out_err) {
                    *out_err = ASTRAL_E_INVALID;
                }
                return false;
            }
            out_devices.push_back(cuda_devices[static_cast<size_t>(idx)]);
            selected_indices.push_back(idx);
        }
    } else if (flags & ASTRAL_GPU_CFG_DEVICE_MASK) {
        if (desc->gpu_device_mask == 0) {
            if (out_err) {
                *out_err = ASTRAL_E_INVALID;
            }
            return false;
        }
        out_devices.reserve(cuda_devices.size() + 1);
        selected_indices.reserve(cuda_devices.size());
        for (size_t i = 0; i < cuda_devices.size(); ++i) {
            if ((desc->gpu_device_mask >> i) & 1ull) {
                out_devices.push_back(cuda_devices[i]);
                selected_indices.push_back(static_cast<int32_t>(i));
            }
        }
        if (out_devices.empty()) {
            if (out_err) {
                *out_err = ASTRAL_E_INVALID;
            }
            return false;
        }
    }

    if (!out_devices.empty()) {
        out_devices.push_back(nullptr);
        params.devices = out_devices.data();
    }

    if (flags & ASTRAL_GPU_CFG_MAIN) {
        if (!selected_indices.empty()) {
            int32_t main_index = -1;
            for (size_t i = 0; i < selected_indices.size(); ++i) {
                if (selected_indices[i] == desc->gpu_main) {
                    main_index = static_cast<int32_t>(i);
                    break;
                }
            }
            if (main_index < 0) {
                if (out_err) {
                    *out_err = ASTRAL_E_INVALID;
                }
                return false;
            }
            params.main_gpu = main_index;
        } else {
            params.main_gpu = desc->gpu_main;
        }
    }

    if (flags & ASTRAL_GPU_CFG_TENSOR_SPLIT) {
        if (desc->gpu_tensor_split == nullptr || desc->gpu_tensor_split_count == 0) {
            if (out_err) {
                *out_err = ASTRAL_E_INVALID;
            }
            return false;
        }
        const size_t max_devices = llama_max_devices();
        out_tensor_split.assign(max_devices, 0.0f);
        const size_t copy_count =
            desc->gpu_tensor_split_count < max_devices ? desc->gpu_tensor_split_count : max_devices;
        if (copy_count > 0) {
            std::memcpy(out_tensor_split.data(), desc->gpu_tensor_split, copy_count * sizeof(float));
            params.tensor_split = out_tensor_split.data();
        }
    }

    return true;
#endif
}

void llama_log_init_from_env() {
    const char* v = std::getenv("ASTRAL_LLAMA_LOG");
    if (v == nullptr || v[0] == '\0') {
        g_llama_log_threshold.store(GGML_LOG_LEVEL_WARN, std::memory_order_relaxed);
        return;
    }

    if (std::strcmp(v, "0") == 0 || std::strcmp(v, "none") == 0) {
        g_llama_log_threshold.store(-1, std::memory_order_relaxed);
        return;
    }
    if (std::strcmp(v, "error") == 0) {
        g_llama_log_threshold.store(GGML_LOG_LEVEL_ERROR, std::memory_order_relaxed);
        return;
    }
    if (std::strcmp(v, "warn") == 0) {
        g_llama_log_threshold.store(GGML_LOG_LEVEL_WARN, std::memory_order_relaxed);
        return;
    }
    if (std::strcmp(v, "info") == 0) {
        g_llama_log_threshold.store(GGML_LOG_LEVEL_INFO, std::memory_order_relaxed);
        return;
    }
    if (std::strcmp(v, "debug") == 0) {
        g_llama_log_threshold.store(GGML_LOG_LEVEL_DEBUG, std::memory_order_relaxed);
        return;
    }

    // Default: keep logs quiet by allowing only warnings/errors.
    g_llama_log_threshold.store(GGML_LOG_LEVEL_WARN, std::memory_order_relaxed);
}

inline void llama_backend_ref() {
    if (g_llama_backend_refs.fetch_add(1, std::memory_order_acq_rel) == 0) {
        llama_log_init_from_env();
        llama_log_set(llama_log_callback, nullptr);
        llama_backend_init();
    }
}

inline void llama_backend_unref() {
    if (g_llama_backend_refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        llama_backend_free();
    }
}

inline uint32_t clamp_threads(uint32_t v, uint32_t fallback) {
    return v > 0 ? v : fallback;
}

inline uint32_t default_threads_fallback() {
    if (astral::core::runtime_initialized()) {
        const uint32_t rt = astral::core::runtime_thread_count();
        if (rt > 0) {
            return rt;
        }
    }
    uint32_t hw = std::thread::hardware_concurrency();
    return hw > 0 ? hw : 4;
}

inline uint32_t clamp_u32(uint32_t v, uint32_t lo, uint32_t hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

#if ASTRAL_ENABLE_MTMD
struct MediaBuffer {
    const uint8_t* data = nullptr;
    size_t size = 0;
    void* owned = nullptr;
    size_t owned_size = 0;
    size_t owned_align = 1;
};

static void media_buffer_free(MediaBuffer* buf) {
    if (buf == nullptr || buf->owned == nullptr) {
        return;
    }
    astral::core::runtime_free(buf->owned, buf->owned_size, buf->owned_align);
    buf->owned = nullptr;
    buf->data = nullptr;
    buf->size = 0;
}

static AstralErr prepare_image_rgb8(const AstralImageDesc* desc, MediaBuffer* out) {
    if (desc == nullptr || out == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (desc->size != sizeof(AstralImageDesc)) {
        return ASTRAL_E_INVALID;
    }
    if (desc->width == 0 || desc->height == 0) {
        return ASTRAL_E_INVALID;
    }
    if (desc->pixels.data == nullptr || desc->pixels.len == 0) {
        return ASTRAL_E_INVALID;
    }

    const uint32_t width = desc->width;
    const uint32_t height = desc->height;

    if (desc->format == ASTRAL_IMAGE_FORMAT_RGB8) {
        const uint32_t row = desc->row_stride != 0 ? desc->row_stride : width * 3u;
        const uint64_t need = static_cast<uint64_t>(row) * static_cast<uint64_t>(height);
        if (desc->pixels.len < need) {
            return ASTRAL_E_INVALID;
        }
        if (row == width * 3u) {
            out->data = desc->pixels.data;
            out->size = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
            return ASTRAL_OK;
        }
        const size_t packed = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
        uint8_t* dst = static_cast<uint8_t*>(astral::core::runtime_alloc(packed, alignof(uint8_t)));
        if (dst == nullptr) {
            return ASTRAL_E_NOMEM;
        }
        const uint8_t* src = desc->pixels.data;
        for (uint32_t y = 0; y < height; ++y) {
            std::memcpy(dst + static_cast<size_t>(y) * width * 3u,
                        src + static_cast<size_t>(y) * row,
                        static_cast<size_t>(width) * 3u);
        }
        out->data = dst;
        out->size = packed;
        out->owned = dst;
        out->owned_size = packed;
        out->owned_align = alignof(uint8_t);
        return ASTRAL_OK;
    }

    if (desc->format == ASTRAL_IMAGE_FORMAT_RGBA8) {
        const uint32_t row = desc->row_stride != 0 ? desc->row_stride : width * 4u;
        const uint64_t need = static_cast<uint64_t>(row) * static_cast<uint64_t>(height);
        if (desc->pixels.len < need) {
            return ASTRAL_E_INVALID;
        }
        const size_t packed = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
        uint8_t* dst = static_cast<uint8_t*>(astral::core::runtime_alloc(packed, alignof(uint8_t)));
        if (dst == nullptr) {
            return ASTRAL_E_NOMEM;
        }
        const uint8_t* src = desc->pixels.data;
        for (uint32_t y = 0; y < height; ++y) {
            const uint8_t* row_src = src + static_cast<size_t>(y) * row;
            uint8_t* row_dst = dst + static_cast<size_t>(y) * width * 3u;
            for (uint32_t x = 0; x < width; ++x) {
                const uint8_t* px = row_src + x * 4u;
                row_dst[x * 3u + 0u] = px[0];
                row_dst[x * 3u + 1u] = px[1];
                row_dst[x * 3u + 2u] = px[2];
            }
        }
        out->data = dst;
        out->size = packed;
        out->owned = dst;
        out->owned_size = packed;
        out->owned_align = alignof(uint8_t);
        return ASTRAL_OK;
    }

    if (desc->format == ASTRAL_IMAGE_FORMAT_RGB_F32) {
        const uint32_t row = desc->row_stride != 0 ? desc->row_stride : width * 3u * sizeof(float);
        const uint64_t need = static_cast<uint64_t>(row) * static_cast<uint64_t>(height);
        if (desc->pixels.len < need) {
            return ASTRAL_E_INVALID;
        }
        const size_t packed = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
        uint8_t* dst = static_cast<uint8_t*>(astral::core::runtime_alloc(packed, alignof(uint8_t)));
        if (dst == nullptr) {
            return ASTRAL_E_NOMEM;
        }
        const uint8_t* src_bytes = desc->pixels.data;
        for (uint32_t y = 0; y < height; ++y) {
            const float* row_src = reinterpret_cast<const float*>(src_bytes + static_cast<size_t>(y) * row);
            uint8_t* row_dst = dst + static_cast<size_t>(y) * width * 3u;
            for (uint32_t x = 0; x < width; ++x) {
                const float* px = row_src + x * 3u;
                for (uint32_t c = 0; c < 3; ++c) {
                    float v = px[c];
                    if (v < 0.0f) v = 0.0f;
                    if (v > 1.0f) v = 1.0f;
                    row_dst[x * 3u + c] = static_cast<uint8_t>(v * 255.0f + 0.5f);
                }
            }
        }
        out->data = dst;
        out->size = packed;
        out->owned = dst;
        out->owned_size = packed;
        out->owned_align = alignof(uint8_t);
        return ASTRAL_OK;
    }

    return ASTRAL_E_UNSUPPORTED;
}

static AstralErr prepare_audio_f32(const AstralAudioDesc* desc, MediaBuffer* out, int32_t expected_rate) {
    if (desc == nullptr || out == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (desc->size != sizeof(AstralAudioDesc)) {
        return ASTRAL_E_INVALID;
    }
    if (desc->channels == 0 || desc->frame_count == 0) {
        return ASTRAL_E_INVALID;
    }
    if (desc->samples.data == nullptr || desc->samples.len == 0) {
        return ASTRAL_E_INVALID;
    }
    if (expected_rate > 0 && desc->sample_rate != static_cast<uint32_t>(expected_rate)) {
        return ASTRAL_E_INVALID;
    }

    const uint64_t frames = desc->frame_count;
    const uint32_t channels = desc->channels;
    const uint64_t total_samples = frames * static_cast<uint64_t>(channels);

    if (desc->format == ASTRAL_AUDIO_FORMAT_F32) {
        const uint64_t need = total_samples * sizeof(float);
        if (desc->samples.len < need) {
            return ASTRAL_E_INVALID;
        }
        if (channels == 1) {
            out->data = desc->samples.data;
            out->size = static_cast<size_t>(frames) * sizeof(float);
            return ASTRAL_OK;
        }

        float* dst = astral::core::runtime_alloc_array<float>(static_cast<uint32_t>(frames));
        if (dst == nullptr) {
            return ASTRAL_E_NOMEM;
        }
        const float* src = reinterpret_cast<const float*>(desc->samples.data);
        for (uint64_t i = 0; i < frames; ++i) {
            double acc = 0.0;
            for (uint32_t c = 0; c < channels; ++c) {
                acc += static_cast<double>(src[i * channels + c]);
            }
            dst[i] = static_cast<float>(acc / static_cast<double>(channels));
        }
        out->data = reinterpret_cast<const uint8_t*>(dst);
        out->size = static_cast<size_t>(frames) * sizeof(float);
        out->owned = dst;
        out->owned_size = static_cast<size_t>(frames) * sizeof(float);
        out->owned_align = alignof(float);
        return ASTRAL_OK;
    }

    if (desc->format == ASTRAL_AUDIO_FORMAT_I16) {
        const uint64_t need = total_samples * sizeof(int16_t);
        if (desc->samples.len < need) {
            return ASTRAL_E_INVALID;
        }
        float* dst = astral::core::runtime_alloc_array<float>(static_cast<uint32_t>(frames));
        if (dst == nullptr) {
            return ASTRAL_E_NOMEM;
        }
        const int16_t* src = reinterpret_cast<const int16_t*>(desc->samples.data);
        for (uint64_t i = 0; i < frames; ++i) {
            double acc = 0.0;
            for (uint32_t c = 0; c < channels; ++c) {
                const int16_t v = src[i * channels + c];
                acc += static_cast<double>(v) / 32768.0;
            }
            dst[i] = static_cast<float>(acc / static_cast<double>(channels));
        }
        out->data = reinterpret_cast<const uint8_t*>(dst);
        out->size = static_cast<size_t>(frames) * sizeof(float);
        out->owned = dst;
        out->owned_size = static_cast<size_t>(frames) * sizeof(float);
        out->owned_align = alignof(float);
        return ASTRAL_OK;
    }

    return ASTRAL_E_UNSUPPORTED;
}

static void mtmd_lock_acquire(std::atomic_flag& flag) {
    while (flag.test_and_set(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

static void mtmd_lock_release(std::atomic_flag& flag) {
    flag.clear(std::memory_order_release);
}
#endif

uint32_t cpu_default_slots_from_env() {
    // Slots are a session-level setting; parsing env vars here is not a hot path.
    const char* v = std::getenv("ASTRAL_LLAMA_MAX_SLOTS");
    if (v == nullptr || v[0] == '\0') {
        return 1;
    }

    uint64_t n = 0;
    for (size_t i = 0; v[i] != '\0'; ++i) {
        const char c = v[i];
        if (c < '0' || c > '9') {
            return 1;
        }
        n = n * 10u + static_cast<uint64_t>(c - '0');
        if (n > 1000000u) {
            return 1;
        }
    }

    return clamp_u32(static_cast<uint32_t>(n), 1u, kCpuMaxSlots);
}

static bool parse_astral_source_id(AstralSpanU8 path, uint64_t* out_id) {
    if (out_id == nullptr) {
        return false;
    }
    *out_id = 0;

    if (path.data == nullptr || path.len == 0) {
        return false;
    }

    static constexpr char kPrefix[] = "astral-src:";
    static constexpr size_t kPrefixLen = sizeof(kPrefix) - 1;
    if (path.len <= kPrefixLen) {
        return false;
    }

    if (std::memcmp(path.data, kPrefix, kPrefixLen) != 0) {
        return false;
    }

    uint64_t id = 0;
    const uint8_t* p = path.data + kPrefixLen;
    const uint8_t* end = path.data + path.len;
    for (; p < end; ++p) {
        const uint8_t c = *p;
        if (c < static_cast<uint8_t>('0') || c > static_cast<uint8_t>('9')) {
            return false;
        }
        const uint64_t next = id * 10u + static_cast<uint64_t>(c - static_cast<uint8_t>('0'));
        if (next < id) {
            return false;
        }
        id = next;
    }

    if (id == 0) {
        return false;
    }

    *out_id = id;
    return true;
}

struct MemReader {
    const uint8_t* data = nullptr;
    uint64_t size = 0;
};

#if ASTRAL_CPU_MEMORY_SOURCE_MMAP
static constexpr uint32_t kCpuModelIoCopyChunkBytes = 64u * 1024u;

static uint32_t ASTRAL_CALL astral_io_read_at_noexcept(const AstralModelIO* io, uint64_t offset, void* dst, uint32_t dst_len) {
    if (io == nullptr || io->read_at == nullptr) {
        return 0;
    }
#if defined(__cpp_exceptions)
    try {
        return io->read_at(io->user, offset, dst, dst_len);
    } catch (...) {
        return 0;
    }
#else
    return io->read_at(io->user, offset, dst, dst_len);
#endif
}

static uint64_t ASTRAL_CALL astral_io_size_noexcept(const AstralModelIO* io) {
    if (io == nullptr || io->size == nullptr) {
        return 0;
    }
#if defined(__cpp_exceptions)
    try {
        return io->size(io->user);
    } catch (...) {
        return 0;
    }
#else
    return io->size(io->user);
#endif
}

#if defined(_WIN32)
static bool write_all_handle(HANDLE h, const uint8_t* data, uint64_t size) {
    if (h == INVALID_HANDLE_VALUE || data == nullptr) {
        return false;
    }

    uint64_t off = 0;
    while (off < size) {
        const uint64_t remaining = size - off;
        const DWORD chunk = remaining > 0x7FFFFFFFu ? 0x7FFFFFFFu : static_cast<DWORD>(remaining);
        DWORD written = 0;
        if (!WriteFile(h, data + off, chunk, &written, nullptr)) {
            return false;
        }
        if (written == 0) {
            return false;
        }
        off += static_cast<uint64_t>(written);
    }
    return true;
}

static bool make_temp_path(char* out_path, size_t cap) {
    if (out_path == nullptr || cap == 0) {
        return false;
    }
    char tmp_dir[MAX_PATH + 1];
    const DWORD n = GetTempPathA(MAX_PATH, tmp_dir);
    if (n == 0 || n > MAX_PATH) {
        return false;
    }
    char tmp_file[MAX_PATH + 1];
    if (GetTempFileNameA(tmp_dir, "ast", 0, tmp_file) == 0) {
        return false;
    }
    const size_t len = std::strlen(tmp_file);
    if (len + 1 > cap) {
        DeleteFileA(tmp_file);
        return false;
    }
    std::memcpy(out_path, tmp_file, len + 1);
    return true;
}
#else
static bool make_temp_path(char* out_path, size_t cap, int* out_fd) {
    if (out_path == nullptr || cap == 0 || out_fd == nullptr) {
        return false;
    }
    char tmpl[] = "/tmp/astral-model-XXXXXX";
    const int fd = ::mkstemp(tmpl);
    if (fd < 0) {
        return false;
    }
    const size_t len = std::strlen(tmpl);
    if (len + 1 > cap) {
        ::close(fd);
        ::unlink(tmpl);
        return false;
    }
    std::memcpy(out_path, tmpl, len + 1);
    *out_fd = fd;
    return true;
}

static bool write_all_fd(int fd, const uint8_t* data, uint64_t size) {
    if (fd < 0 || data == nullptr) {
        return false;
    }

    uint64_t off = 0;
    while (off < size) {
        const size_t chunk = (size - off) > 0x7FFFFFFFu ? 0x7FFFFFFFu : static_cast<size_t>(size - off);
        const ssize_t n = ::write(fd, data + off, chunk);
        if (n <= 0) {
            return false;
        }
        off += static_cast<uint64_t>(n);
    }
    return true;
}
#endif

static llama_model* try_load_memory_source_with_mmap(const uint8_t* data, uint64_t size, llama_model_params model_params) {
    if (data == nullptr || size == 0) {
        return nullptr;
    }

    if (!llama_supports_mmap()) {
        return nullptr;
    }

    char path[512]{};

#if defined(_WIN32)
    if (!make_temp_path(path, sizeof(path))) {
        return nullptr;
    }
    HANDLE h = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        DeleteFileA(path);
        return nullptr;
    }
    const bool ok = write_all_handle(h, data, size);
    CloseHandle(h);
    if (!ok) {
        DeleteFileA(path);
        return nullptr;
    }

    ASTRAL_ZONE_N("astral.cpu.llama_model_load_from_file.memory_tmp");
    llama_model* model = llama_model_load_from_file(path, model_params);
    DeleteFileA(path);
    return model;
#else
    int fd = -1;
    if (!make_temp_path(path, sizeof(path), &fd)) {
        return nullptr;
    }
    const bool ok = write_all_fd(fd, data, size);
    ::close(fd);
    if (!ok) {
        ::unlink(path);
        return nullptr;
    }

    ASTRAL_ZONE_N("astral.cpu.llama_model_load_from_file.memory_tmp");
    llama_model* model = llama_model_load_from_file(path, model_params);
    ::unlink(path);
    return model;
#endif
}

static llama_model* try_load_io_source_with_mmap(const AstralModelIO* src_io, llama_model_params model_params) {
    if (src_io == nullptr) {
        return nullptr;
    }
    if (!llama_supports_mmap()) {
        return nullptr;
    }

    const uint64_t size = astral_io_size_noexcept(src_io);
    if (size == 0) {
        return nullptr;
    }

    char path[512]{};

#if defined(_WIN32)
    if (!make_temp_path(path, sizeof(path))) {
        return nullptr;
    }
    HANDLE h = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        DeleteFileA(path);
        return nullptr;
    }

    uint8_t buf[kCpuModelIoCopyChunkBytes];
    uint64_t off = 0;
    bool ok = true;
    while (off < size) {
        const uint64_t remaining = size - off;
        const uint32_t want = remaining > sizeof(buf) ? static_cast<uint32_t>(sizeof(buf)) : static_cast<uint32_t>(remaining);
        const uint32_t got = astral_io_read_at_noexcept(src_io, off, buf, want);
        if (got == 0) {
            ok = false;
            break;
        }
        ok = write_all_handle(h, buf, got);
        if (!ok) {
            break;
        }
        off += got;
    }

    CloseHandle(h);
    if (!ok) {
        DeleteFileA(path);
        return nullptr;
    }

    ASTRAL_ZONE_N("astral.cpu.llama_model_load_from_file.io_tmp");
    llama_model* model = llama_model_load_from_file(path, model_params);
    DeleteFileA(path);
    return model;
#else
    int fd = -1;
    if (!make_temp_path(path, sizeof(path), &fd)) {
        return nullptr;
    }

    uint8_t buf[kCpuModelIoCopyChunkBytes];
    uint64_t off = 0;
    bool ok = true;
    while (off < size) {
        const uint64_t remaining = size - off;
        const uint32_t want = remaining > sizeof(buf) ? static_cast<uint32_t>(sizeof(buf)) : static_cast<uint32_t>(remaining);
        const uint32_t got = astral_io_read_at_noexcept(src_io, off, buf, want);
        if (got == 0) {
            ok = false;
            break;
        }
        ok = write_all_fd(fd, buf, got);
        if (!ok) {
            break;
        }
        off += got;
    }

    ::close(fd);
    if (!ok) {
        ::unlink(path);
        return nullptr;
    }

    ASTRAL_ZONE_N("astral.cpu.llama_model_load_from_file.io_tmp");
    llama_model* model = llama_model_load_from_file(path, model_params);
    ::unlink(path);
    return model;
#endif
}
#endif

void* cpu_model_load(const AstralModelDesc* desc, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }

    if (desc == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    // Reference-counted llama.cpp backend init sits outside decode hot paths.
    llama_backend_ref();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = static_cast<int32_t>(desc->gpu_layers);
    // Prefer mmap-backed weights for faster startup and lower RSS (when supported by the OS).
    model_params.use_mmap = true;
    model_params.use_mlock = false;

    std::vector<ggml_backend_dev_t> gpu_devices;
    std::vector<float> gpu_tensor_split;
    const AstralModelDesc* desc_cfg = astral::core::model_load_desc();
    if (!apply_gpu_config(desc_cfg, model_params, gpu_devices, gpu_tensor_split, out_err)) {
        llama_backend_unref();
        return nullptr;
    }

    llama_model* model = nullptr;

    uint64_t src_id = 0;
    if (parse_astral_source_id(desc->model_path, &src_id)) {
        astral::core::ModelSource src{};
        if (!astral::core::model_source_take(src_id, &src)) {
            llama_backend_unref();
            *out_err = ASTRAL_E_INVALID;
            return nullptr;
        }

        if (src.kind == ASTRAL_MODEL_SOURCE_MEMORY) {
#if ASTRAL_CPU_MEMORY_SOURCE_MMAP
            model = try_load_memory_source_with_mmap(src.bytes.data, src.bytes.len, model_params);
#else
            model = nullptr;
#endif
        } else if (src.kind == ASTRAL_MODEL_SOURCE_IO) {
#if ASTRAL_CPU_MEMORY_SOURCE_MMAP
            model = try_load_io_source_with_mmap(&src.io, model_params);
#else
            model = nullptr;
#endif
        } else {
            llama_backend_unref();
            *out_err = ASTRAL_E_UNSUPPORTED;
            return nullptr;
        }

        if (model == nullptr) {
            llama_backend_unref();
            *out_err = ASTRAL_E_UNSUPPORTED;
            return nullptr;
        }
    } else {
        if (desc->model_path.data == nullptr || desc->model_path.len == 0) {
            llama_backend_unref();
            *out_err = ASTRAL_E_INVALID;
            return nullptr;
        }

        // llama_model_load_from_file expects a NUL-terminated path.
        const size_t path_len = static_cast<size_t>(desc->model_path.len);
        char* path = static_cast<char*>(astral::core::runtime_alloc(path_len + 1, 1));
        if (path == nullptr) {
            llama_backend_unref();
            *out_err = ASTRAL_E_NOMEM;
            return nullptr;
        }
        std::memcpy(path, desc->model_path.data, path_len);
        path[path_len] = '\0';

        ASTRAL_ZONE_N("astral.cpu.llama_model_load_from_file");
        model = llama_model_load_from_file(path, model_params);
        astral::core::runtime_free(path, path_len + 1, 1);
    }

    if (model == nullptr) {
        llama_backend_unref();
        *out_err = ASTRAL_E_BACKEND;
        return nullptr;
    }

    CpuModel* cpu_model = astral::core::runtime_new<CpuModel>();
    if (cpu_model == nullptr) {
        llama_model_free(model);
        llama_backend_unref();
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    cpu_model->gpu_devices = std::move(gpu_devices);
    cpu_model->gpu_tensor_split = std::move(gpu_tensor_split);

    cpu_model->model = model;
    cpu_model->vocab = llama_model_get_vocab(model);
    if (cpu_model->vocab == nullptr) {
        llama_model_free(model);
        astral::core::runtime_delete(cpu_model);
        llama_backend_unref();
        *out_err = ASTRAL_E_BACKEND;
        return nullptr;
    }

    cpu_model->n_ctx = desc->n_ctx;
    if (cpu_model->n_ctx == 0) {
        const int32_t n_ctx_train = llama_model_n_ctx_train(model);
        if (n_ctx_train > 0) {
            cpu_model->n_ctx = static_cast<uint32_t>(n_ctx_train);
        }
    }

    if (cpu_model->n_ctx == 0) {
        cpu_model->n_ctx = 2048;
    }

    cpu_model->n_batch = desc->n_batch > 0 ? desc->n_batch : 512;
    if (cpu_model->n_batch > cpu_model->n_ctx) {
        cpu_model->n_batch = cpu_model->n_ctx;
    }
    cpu_model->n_threads = desc->n_threads; // 0 means default
    cpu_model->n_threads_batch = desc->n_threads;
    cpu_model->embeddings_only = desc->embeddings_only;

    cpu_model->vocab_size = static_cast<uint32_t>(llama_vocab_n_tokens(cpu_model->vocab));
    cpu_model->token_bos = static_cast<int32_t>(llama_vocab_bos(cpu_model->vocab));
    cpu_model->token_eos = static_cast<int32_t>(llama_vocab_eos(cpu_model->vocab));
    const int32_t n_embd = llama_model_n_embd(model);
    cpu_model->n_embd = n_embd > 0 ? static_cast<uint32_t>(n_embd) : 0;

#if ASTRAL_ENABLE_MTMD
    cpu_model->mtmd = nullptr;
    cpu_model->media_initialized = 0;
    cpu_model->supports_vision = 0;
    cpu_model->supports_audio = 0;
    cpu_model->image_min_tokens = 0;
    cpu_model->image_max_tokens = 0;
    cpu_model->audio_sample_rate = 0;
    cpu_model->mtmd_lock.clear(std::memory_order_release);
#endif

    *out_err = ASTRAL_OK;
    return cpu_model;
}

void cpu_model_unload(void* model_ctx) {
    if (model_ctx == nullptr) {
        return;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);
#if ASTRAL_ENABLE_MTMD
    if (model->mtmd) {
        mtmd_free(model->mtmd);
        model->mtmd = nullptr;
    }
#endif
    if (model->model) {
        llama_model_free(model->model);
    }
    astral::core::runtime_delete(model);
    llama_backend_unref();
}

AstralErr cpu_tokenize(void* model_ctx, AstralSpanU8 text,
                       int32_t* out_tokens, uint32_t max_tokens,
                       uint8_t add_special, uint8_t parse_special,
                       uint32_t* out_count) {
    ASTRAL_ZONE_MICRO_N("astral.cpu.tokenize");
    if (model_ctx == nullptr || out_count == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);
    const llama_vocab* vocab = model->vocab;
    if (vocab == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    const char* input = text.data ? reinterpret_cast<const char*>(text.data) : "";
    const int32_t input_len = static_cast<int32_t>(text.len);

    const int32_t n_tokens = llama_tokenize(
        vocab,
        input,
        input_len,
        reinterpret_cast<llama_token*>(out_tokens),
        static_cast<int32_t>(max_tokens),
        add_special != 0,
        parse_special != 0
    );

    if (n_tokens < 0) {
        *out_count = static_cast<uint32_t>(-n_tokens);
        return out_tokens == nullptr ? ASTRAL_OK : ASTRAL_E_NOMEM;
    }

    *out_count = static_cast<uint32_t>(n_tokens);
    return ASTRAL_OK;
}

AstralErr cpu_detokenize(void* model_ctx, const int32_t* tokens, uint32_t count,
                         AstralMutSpanU8 out_text, uint32_t* out_len) {
    ASTRAL_ZONE_MICRO_N("astral.cpu.detokenize");
    if (model_ctx == nullptr || (tokens == nullptr && count != 0) || out_len == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);
    const llama_vocab* vocab = model->vocab;
    if (vocab == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    uint32_t offset = 0;
    if (out_text.data == nullptr) {
      char count_buf[1]{};
      for (uint32_t i = 0; i < count; ++i) {
        const int32_t written =
            llama_token_to_piece(vocab, static_cast<llama_token>(tokens[i]), count_buf, 0,
                                 0,    // lstrip (don't strip leading space)
                                 false // special (don't render special tokens)
            );
        offset += static_cast<uint32_t>(-written);
      }
      *out_len = offset;
      return ASTRAL_OK;
    }

    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t space = out_text.len - offset;
      const int32_t written = llama_token_to_piece(vocab, static_cast<llama_token>(tokens[i]),
                                                   reinterpret_cast<char*>(out_text.data + offset),
                                                   static_cast<int32_t>(space),
                                                   0,    // lstrip (don't strip leading space)
                                                   false // special (don't render special tokens)
      );
      if (written < 0) {
        offset += static_cast<uint32_t>(-written);
        *out_len = offset;
        return ASTRAL_E_NOMEM;
      }
      offset += static_cast<uint32_t>(written);
    }

    *out_len = offset;
    return ASTRAL_OK;
}

AstralErr cpu_model_info(void* model_ctx, uint32_t* out_vocab_size, uint32_t* out_ctx_size) {
    if (model_ctx == nullptr || out_vocab_size == nullptr || out_ctx_size == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);
    *out_vocab_size = model->vocab_size;
    *out_ctx_size = model->n_ctx;
    return ASTRAL_OK;
}

AstralErr cpu_model_special_tokens(void* model_ctx, int32_t* out_bos, int32_t* out_eos) {
    if (model_ctx == nullptr || out_bos == nullptr || out_eos == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);
    *out_bos = model->token_bos;
    *out_eos = model->token_eos;
    return ASTRAL_OK;
}

AstralErr cpu_model_embedding_dim(void* model_ctx, uint32_t* out_dim) {
    if (model_ctx == nullptr || out_dim == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);
    if (model->n_embd == 0) {
        return ASTRAL_E_BACKEND;
    }

    *out_dim = model->n_embd;
    return ASTRAL_OK;
}

AstralErr cpu_model_media_init(void* model_ctx, const AstralModelMediaDesc* desc) {
#if !ASTRAL_ENABLE_MTMD
    (void)model_ctx;
    (void)desc;
    return ASTRAL_E_UNSUPPORTED;
#else
    if (model_ctx == nullptr || desc == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (desc->size != sizeof(AstralModelMediaDesc)) {
        return ASTRAL_E_INVALID;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);
    if (model->media_initialized != 0) {
        return ASTRAL_E_STATE;
    }
    if (desc->source_kind != ASTRAL_MODEL_SOURCE_PATH) {
        return ASTRAL_E_UNSUPPORTED;
    }
    if (desc->media_path.data == nullptr || desc->media_path.len == 0) {
        return ASTRAL_E_INVALID;
    }

    const size_t path_len = static_cast<size_t>(desc->media_path.len);
    char* path = static_cast<char*>(astral::core::runtime_alloc(path_len + 1, 1));
    if (path == nullptr) {
        return ASTRAL_E_NOMEM;
    }
    std::memcpy(path, desc->media_path.data, path_len);
    path[path_len] = '\0';

    mtmd_context_params params = mtmd_context_params_default();
    const bool route_gpu =
        ((desc->flags & ASTRAL_MEDIA_FLAG_USE_GPU) != 0) || desc->gpu_route_flags != 0 || desc->gpu_device_mask != 0;
    params.use_gpu = route_gpu;
    params.warmup = (desc->flags & ASTRAL_MEDIA_FLAG_WARMUP) != 0;
    params.n_threads = static_cast<int>(clamp_threads(model->n_threads, default_threads_fallback()));
    params.image_min_tokens = static_cast<int>(desc->image_min_tokens);
    params.image_max_tokens = static_cast<int>(desc->image_max_tokens);

    mtmd_context* ctx = mtmd_init_from_file(path, model->model, params);
    astral::core::runtime_free(path, path_len + 1, 1);

    if (ctx == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    model->mtmd = ctx;
    model->media_initialized = 1;
    model->supports_vision = mtmd_support_vision(ctx) ? 1 : 0;
    model->supports_audio = mtmd_support_audio(ctx) ? 1 : 0;
    model->audio_sample_rate = mtmd_get_audio_sample_rate(ctx);
    model->image_min_tokens = desc->image_min_tokens;
    model->image_max_tokens = desc->image_max_tokens;
    return ASTRAL_OK;
#endif
}

AstralErr cpu_model_media_info(void* model_ctx, AstralMediaInfo* out_info) {
#if !ASTRAL_ENABLE_MTMD
    (void)model_ctx;
    (void)out_info;
    return ASTRAL_E_UNSUPPORTED;
#else
    if (model_ctx == nullptr || out_info == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (out_info->size != sizeof(AstralMediaInfo)) {
        return ASTRAL_E_INVALID;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);
    if (model->mtmd == nullptr) {
        out_info->supports_image = 0;
        out_info->supports_audio = 0;
        out_info->audio_sample_rate = 0;
        out_info->image_min_tokens = 0;
        out_info->image_max_tokens = 0;
        return ASTRAL_OK;
    }

    out_info->supports_image = model->supports_vision != 0 ? 1u : 0u;
    out_info->supports_audio = model->supports_audio != 0 ? 1u : 0u;
    out_info->audio_sample_rate = model->audio_sample_rate > 0 ? static_cast<uint32_t>(model->audio_sample_rate) : 0u;
    out_info->image_min_tokens = model->image_min_tokens;
    out_info->image_max_tokens = model->image_max_tokens;
    return ASTRAL_OK;
#endif
}

void* cpu_session_create_ex(void* model_ctx, const AstralSessionDesc* desc, uint32_t max_slots, AstralErr* out_err);

void* cpu_session_create(void* model_ctx, const AstralSessionDesc* desc, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }

    if (model_ctx == nullptr || desc == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    return cpu_session_create_ex(model_ctx, desc, cpu_default_slots_from_env(), out_err);
}

void* cpu_session_create_ex(void* model_ctx, const AstralSessionDesc* desc, uint32_t max_slots, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }

    if (model_ctx == nullptr || desc == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    const uint32_t slots = max_slots == 0 ? 1u : (max_slots > kCpuMaxSlots ? kCpuMaxSlots : max_slots);

    CpuModel* model = static_cast<CpuModel*>(model_ctx);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = model->n_ctx;
    ctx_params.n_batch = model->n_batch;
    ctx_params.n_ubatch = ctx_params.n_batch;
    const uint32_t threads_default = default_threads_fallback();
    ctx_params.n_threads = static_cast<int32_t>(clamp_threads(model->n_threads, threads_default));
    ctx_params.n_threads_batch =
        static_cast<int32_t>(clamp_threads(model->n_threads_batch, static_cast<uint32_t>(ctx_params.n_threads)));
    ctx_params.n_seq_max = static_cast<int32_t>(slots);

    llama_context* ctx = llama_init_from_model(model->model, ctx_params);
    if (ctx == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    CpuSession* session = astral::core::runtime_new<CpuSession>();
    if (session == nullptr) {
        llama_free(ctx);
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    session->model = model;
    session->ctx = ctx;
    session->n_batch = model->n_batch;
    session->n_slots = static_cast<uint32_t>(ctx_params.n_seq_max);
    session->active_slot = 0;
    session->last_batch_outputs = 0;
    session->last_batch_token_count = 0;
    session->adapter_count = 0;

    for (uint32_t i = 0; i < kCpuMaxSlots; ++i) {
        session->slot_pos[i] = 0;
        session->slot_last_token[i] = 0;
        session->slot_has_logits[i] = false;
        session->slot_last_token_valid[i] = false;
        session->grammar[i] = nullptr;
    }
    for (uint32_t i = 0; i < ASTRAL_SESSION_ADAPTERS_MAX; ++i) {
        session->adapters[i] = nullptr;
        session->adapter_scales[i] = 0.0f;
    }

    // Preallocate batch scratch.
    const uint32_t cap = session->n_batch > 0 ? session->n_batch : 1;
    session->batch_token_storage = astral::core::runtime_alloc_array<int32_t>(cap);
    session->batch_output_token_index = astral::core::runtime_alloc_array<int32_t>(cap);
    session->batch_pos = astral::core::runtime_alloc_array<int32_t>(cap);
    session->batch_n_seq_id = astral::core::runtime_alloc_array<int32_t>(cap);
    session->batch_seq_id_storage = astral::core::runtime_alloc_array<int32_t>(cap);
    session->batch_seq_id_ptrs = astral::core::runtime_alloc_array<int32_t*>(cap);
    session->batch_logits = astral::core::runtime_alloc_array<int8_t>(cap);

    if (session->batch_token_storage == nullptr || session->batch_output_token_index == nullptr || session->batch_pos == nullptr ||
        session->batch_n_seq_id == nullptr || session->batch_seq_id_storage == nullptr || session->batch_seq_id_ptrs == nullptr ||
        session->batch_logits == nullptr) {
        astral::core::runtime_free_array(session->batch_token_storage, cap);
        astral::core::runtime_free_array(session->batch_output_token_index, cap);
        astral::core::runtime_free_array(session->batch_pos, cap);
        astral::core::runtime_free_array(session->batch_n_seq_id, cap);
        astral::core::runtime_free_array(session->batch_seq_id_storage, cap);
        astral::core::runtime_free_array(session->batch_seq_id_ptrs, cap);
        astral::core::runtime_free_array(session->batch_logits, cap);
        astral::core::runtime_delete(session);
        llama_free(ctx);
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    for (uint32_t i = 0; i < cap; ++i) {
      session->batch_n_seq_id[i] = 1;
      session->batch_seq_id_ptrs[i] = &session->batch_seq_id_storage[i];
    }

    *out_err = ASTRAL_OK;
    return session;
}

void cpu_session_destroy(void* session_ctx) {
    if (session_ctx == nullptr) {
        return;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    for (uint32_t i = 0; i < kCpuMaxSlots; ++i) {
        if (session->grammar[i] != nullptr) {
            llama_sampler_free(session->grammar[i]);
            session->grammar[i] = nullptr;
        }
    }
    const uint32_t cap = session->n_batch > 0 ? session->n_batch : 1;
    astral::core::runtime_free_array(session->batch_token_storage, cap);
    astral::core::runtime_free_array(session->batch_output_token_index, cap);
    astral::core::runtime_free_array(session->batch_pos, cap);
    astral::core::runtime_free_array(session->batch_n_seq_id, cap);
    astral::core::runtime_free_array(session->batch_seq_id_storage, cap);
    astral::core::runtime_free_array(session->batch_seq_id_ptrs, cap);
    astral::core::runtime_free_array(session->batch_logits, cap);
    if (session->ctx) {
        llama_free(session->ctx);
    }
    astral::core::runtime_delete(session);
}

AstralErr cpu_session_reset(void* session_ctx) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->ctx == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    // Clear KV/memory for all sequences (provider-side state).
    // This keeps the context allocated and avoids session re-creation overhead.
    llama_memory_t mem = llama_get_memory(session->ctx);
    if (mem == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    llama_memory_clear(mem, /*data=*/true);
    for (uint32_t i = 0; i < session->n_slots; ++i) {
        session->slot_pos[i] = 0;
        session->slot_has_logits[i] = false;
        if (session->grammar[i] != nullptr) {
            llama_sampler_reset(session->grammar[i]);
        }
    }
    return ASTRAL_OK;
}

AstralErr cpu_session_feed(void* session_ctx, const int32_t* tokens, uint32_t count) {
    ASTRAL_ZONE_MICRO_N("astral.cpu.session_feed");
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (count > 0 && tokens == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);

    if (session->active_slot >= session->n_slots) {
        return ASTRAL_E_STATE;
    }

    const uint32_t batch_cap = session->n_batch > 0 ? session->n_batch : count;
    const uint32_t batch_size = batch_cap > 0 ? batch_cap : count;
    int32_t slot_pos = session->slot_pos[session->active_slot];
    for (uint32_t i = 0; i < count; i += batch_size) {
        const uint32_t n = (count - i) < batch_size ? (count - i) : batch_size;

        llama_token* chunk = const_cast<llama_token*>(
            reinterpret_cast<const llama_token*>(tokens + i)
        );

        llama_batch batch{};
        batch.n_tokens = static_cast<int32_t>(n);
        batch.token = chunk;
        batch.embd = nullptr;
        batch.pos = reinterpret_cast<llama_pos*>(session->batch_pos);
        batch.n_seq_id = session->batch_n_seq_id;
        batch.seq_id = reinterpret_cast<llama_seq_id**>(session->batch_seq_id_ptrs);
        batch.logits = session->batch_logits;

        for (uint32_t j = 0; j < n; ++j) {
            session->batch_pos[j] = slot_pos + static_cast<int32_t>(j);
            session->batch_seq_id_storage[j] = static_cast<int32_t>(session->active_slot);
            session->batch_logits[j] = 0;
        }
        session->batch_logits[n - 1u] = 1;

        ASTRAL_ZONE_MICRO_N("astral.cpu.llama_decode");
        const int32_t result = llama_decode(session->ctx, batch);
        if (result != 0) {
            return ASTRAL_E_BACKEND;
        }

        slot_pos += static_cast<int32_t>(n);
    }

    if (count > 0) {
        session->slot_pos[session->active_slot] = slot_pos;
        session->slot_last_token[session->active_slot] = static_cast<int32_t>(tokens[count - 1u]);
        session->slot_has_logits[session->active_slot] = true;
        session->slot_last_token_valid[session->active_slot] = true;
    }

    return ASTRAL_OK;
}

AstralErr cpu_session_feed_image(void* session_ctx, const AstralImageDesc* image, uint8_t finalize) {
    (void)finalize;
#if !ASTRAL_ENABLE_MTMD
    (void)session_ctx;
    (void)image;
    return ASTRAL_E_UNSUPPORTED;
#else
    if (session_ctx == nullptr || image == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->model == nullptr || session->ctx == nullptr) {
        return ASTRAL_E_BACKEND;
    }
    CpuModel* model = session->model;
    if (model->mtmd == nullptr || model->supports_vision == 0) {
        return ASTRAL_E_UNSUPPORTED;
    }
    if (session->active_slot >= session->n_slots) {
        return ASTRAL_E_STATE;
    }

    MediaBuffer buf{};
    const AstralErr prep = prepare_image_rgb8(image, &buf);
    if (prep != ASTRAL_OK) {
        return prep;
    }

    mtmd_bitmap* bmp = mtmd_bitmap_init(image->width, image->height, buf.data);
    media_buffer_free(&buf);
    if (bmp == nullptr) {
        return ASTRAL_E_NOMEM;
    }

    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    if (chunks == nullptr) {
        mtmd_bitmap_free(bmp);
        return ASTRAL_E_NOMEM;
    }

    const mtmd_bitmap* bitmaps[1] = { bmp };
    mtmd_input_text text{};
    text.text = mtmd_default_marker();
    text.add_special = session->slot_pos[session->active_slot] == 0 ? true : false;
    text.parse_special = false;

    int32_t rc = 0;
    mtmd_lock_acquire(model->mtmd_lock);
    rc = mtmd_tokenize(model->mtmd, chunks, &text, bitmaps, 1);
    if (rc == 0) {
        llama_pos n_past = session->slot_pos[session->active_slot];
        llama_pos new_n_past = n_past;
        rc = mtmd_helper_eval_chunks(
            model->mtmd,
            session->ctx,
            chunks,
            n_past,
            static_cast<llama_seq_id>(session->active_slot),
            static_cast<int32_t>(session->n_batch),
            true,
            &new_n_past
        );
        if (rc == 0) {
            session->slot_pos[session->active_slot] = static_cast<int32_t>(new_n_past);
            session->slot_has_logits[session->active_slot] = true;
        }
    }
    mtmd_lock_release(model->mtmd_lock);

    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);

    return rc == 0 ? ASTRAL_OK : ASTRAL_E_BACKEND;
#endif
}

AstralErr cpu_session_feed_audio(void* session_ctx, const AstralAudioDesc* audio, uint8_t finalize) {
    (void)finalize;
#if !ASTRAL_ENABLE_MTMD
    (void)session_ctx;
    (void)audio;
    return ASTRAL_E_UNSUPPORTED;
#else
    if (session_ctx == nullptr || audio == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->model == nullptr || session->ctx == nullptr) {
        return ASTRAL_E_BACKEND;
    }
    CpuModel* model = session->model;
    if (model->mtmd == nullptr || model->supports_audio == 0) {
        return ASTRAL_E_UNSUPPORTED;
    }
    if (session->active_slot >= session->n_slots) {
        return ASTRAL_E_STATE;
    }

    MediaBuffer buf{};
    const AstralErr prep = prepare_audio_f32(audio, &buf, model->audio_sample_rate);
    if (prep != ASTRAL_OK) {
        return prep;
    }

    const size_t sample_count = buf.size / sizeof(float);
    mtmd_bitmap* bmp = mtmd_bitmap_init_from_audio(sample_count, reinterpret_cast<const float*>(buf.data));
    media_buffer_free(&buf);
    if (bmp == nullptr) {
        return ASTRAL_E_NOMEM;
    }

    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    if (chunks == nullptr) {
        mtmd_bitmap_free(bmp);
        return ASTRAL_E_NOMEM;
    }

    const mtmd_bitmap* bitmaps[1] = { bmp };
    mtmd_input_text text{};
    text.text = mtmd_default_marker();
    text.add_special = session->slot_pos[session->active_slot] == 0 ? true : false;
    text.parse_special = false;

    int32_t rc = 0;
    mtmd_lock_acquire(model->mtmd_lock);
    rc = mtmd_tokenize(model->mtmd, chunks, &text, bitmaps, 1);
    if (rc == 0) {
        llama_pos n_past = session->slot_pos[session->active_slot];
        llama_pos new_n_past = n_past;
        rc = mtmd_helper_eval_chunks(
            model->mtmd,
            session->ctx,
            chunks,
            n_past,
            static_cast<llama_seq_id>(session->active_slot),
            static_cast<int32_t>(session->n_batch),
            true,
            &new_n_past
        );
        if (rc == 0) {
            session->slot_pos[session->active_slot] = static_cast<int32_t>(new_n_past);
            session->slot_has_logits[session->active_slot] = true;
        }
    }
    mtmd_lock_release(model->mtmd_lock);

    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);

    return rc == 0 ? ASTRAL_OK : ASTRAL_E_BACKEND;
#endif
}

AstralErr cpu_session_logits(void* session_ctx, BackendLogitsView* out_view) {
    ASTRAL_ZONE_MICRO_N("astral.cpu.session_logits");
    if (session_ctx == nullptr || out_view == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->active_slot >= session->n_slots) {
        return ASTRAL_E_STATE;
    }
    if (!session->slot_has_logits[session->active_slot]) {
        return ASTRAL_E_STATE;
    }

    float* logits = llama_get_logits_ith(session->ctx, -1);
    if (logits == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    out_view->logits = logits;
    out_view->vocab_size = session->model ? session->model->vocab_size : 0;
    if (out_view->vocab_size == 0) {
        return ASTRAL_E_BACKEND;
    }

    return ASTRAL_OK;
}

AstralErr cpu_session_accept(void* session_ctx, int32_t token) {
    ASTRAL_ZONE_MICRO_N("astral.cpu.session_accept");
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->active_slot >= session->n_slots) {
        return ASTRAL_E_STATE;
    }
    if (!session->slot_has_logits[session->active_slot]) {
        return ASTRAL_E_STATE;
    }

    llama_token t = static_cast<llama_token>(token);
    llama_batch batch{};
    batch.n_tokens = 1;
    batch.token = &t;
    batch.embd = nullptr;
    batch.pos = reinterpret_cast<llama_pos*>(session->batch_pos);
    batch.n_seq_id = session->batch_n_seq_id;
    batch.seq_id = reinterpret_cast<llama_seq_id**>(session->batch_seq_id_ptrs);
    batch.logits = session->batch_logits;

    session->batch_pos[0] = session->slot_pos[session->active_slot];
    session->batch_seq_id_storage[0] = static_cast<int32_t>(session->active_slot);
    session->batch_logits[0] = 1;

    ASTRAL_ZONE_MICRO_N("astral.cpu.llama_decode");
    const int32_t result = llama_decode(session->ctx, batch);
    if (result != 0) {
        return ASTRAL_E_BACKEND;
    }

    if (session->grammar[session->active_slot] != nullptr) {
        llama_sampler_accept(session->grammar[session->active_slot], t);
    }

    session->slot_pos[session->active_slot] += 1;
    session->slot_last_token[session->active_slot] = static_cast<int32_t>(t);
    session->slot_has_logits[session->active_slot] = true;
    session->slot_last_token_valid[session->active_slot] = true;
    return ASTRAL_OK;
}

// -----------------------------------------------------------------------------
// Continuous batching (multi-slot eval)
// -----------------------------------------------------------------------------

AstralErr cpu_session_batch_eval(void* session_ctx,
                                 const AstralBackendBatchToken* tokens,
                                 uint32_t token_count,
                                 uint32_t* out_output_count) {
    ASTRAL_ZONE_MICRO_N("astral.cpu.session_batch_eval");
    if (session_ctx == nullptr || out_output_count == nullptr) {
        return ASTRAL_E_INVALID;
    }

    if (token_count > 0 && tokens == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->ctx == nullptr || session->model == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    const uint32_t cap = session->n_batch > 0 ? session->n_batch : 1;
    if (token_count > cap) {
        return ASTRAL_E_NOMEM;
    }

    uint32_t outputs = 0;
    for (uint32_t i = 0; i < token_count; ++i) {
        const AstralBackendBatchToken& t = tokens[i];
        if (t.slot_id >= session->n_slots) {
            return ASTRAL_E_INVALID;
        }
        if (t.pos > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            return ASTRAL_E_INVALID;
        }

        session->batch_token_storage[i] = t.token;
        session->batch_pos[i] = static_cast<int32_t>(t.pos);
        session->batch_seq_id_storage[i] = static_cast<int32_t>(t.slot_id);
        session->batch_logits[i] = t.want_logits != 0 ? 1 : 0;

        if (t.want_logits != 0) {
            session->batch_output_token_index[outputs] = static_cast<int32_t>(i);
            session->slot_last_token[t.slot_id] = t.token;
            session->slot_last_token_valid[t.slot_id] = true;
            outputs += 1u;
        }

        // Track the highest consumed slot position used by state save/load callers.
        const uint32_t next = t.pos + 1u;
        if (next > static_cast<uint32_t>(session->slot_pos[t.slot_id])) {
            session->slot_pos[t.slot_id] = static_cast<int32_t>(next);
        }
        session->slot_has_logits[t.slot_id] = false;
    }

    llama_batch batch{};
    batch.n_tokens = static_cast<int32_t>(token_count);
    batch.token = reinterpret_cast<llama_token*>(session->batch_token_storage);
    batch.embd = nullptr;
    batch.pos = reinterpret_cast<llama_pos*>(session->batch_pos);
    batch.n_seq_id = session->batch_n_seq_id;
    batch.seq_id = reinterpret_cast<llama_seq_id**>(session->batch_seq_id_ptrs);
    batch.logits = session->batch_logits;

    {
        ASTRAL_ZONE_MICRO_N("astral.cpu.llama_decode_batch");
        const int32_t result = llama_decode(session->ctx, batch);
        if (result != 0) {
            return ASTRAL_E_BACKEND;
        }
    }

    session->last_batch_outputs = outputs;
    session->last_batch_token_count = token_count;
    *out_output_count = outputs;
    return ASTRAL_OK;
}

AstralErr cpu_session_batch_logits(void* session_ctx, uint32_t output_index, BackendLogitsView* out_view) {
    ASTRAL_ZONE_MICRO_N("astral.cpu.session_batch_logits");
    if (session_ctx == nullptr || out_view == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->ctx == nullptr || session->model == nullptr) {
        return ASTRAL_E_BACKEND;
    }
    if (output_index >= session->last_batch_outputs) {
        return ASTRAL_E_INVALID;
    }

    const int32_t token_index = session->batch_output_token_index[output_index];
    if (token_index < 0 || static_cast<uint32_t>(token_index) >= session->last_batch_token_count) {
        return ASTRAL_E_BACKEND;
    }

    float* logits = llama_get_logits_ith(session->ctx, token_index);
    if (logits == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    out_view->logits = logits;
    out_view->vocab_size = session->model->vocab_size;
    if (out_view->vocab_size == 0) {
        return ASTRAL_E_BACKEND;
    }

    return ASTRAL_OK;
}

AstralErr cpu_session_slot_reset(void* session_ctx, uint32_t slot_id) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->ctx == nullptr) {
        return ASTRAL_E_BACKEND;
    }
    if (slot_id >= session->n_slots) {
        return ASTRAL_E_INVALID;
    }

    llama_memory_t mem = llama_get_memory(session->ctx);
    if (mem == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    if (!llama_memory_seq_rm(mem, static_cast<llama_seq_id>(slot_id), -1, -1)) {
        return ASTRAL_E_BACKEND;
    }

    session->slot_pos[slot_id] = 0;
    session->slot_has_logits[slot_id] = false;
    if (session->grammar[slot_id] != nullptr) {
        llama_sampler_reset(session->grammar[slot_id]);
    }
    return ASTRAL_OK;
}

// -----------------------------------------------------------------------------
// Optional generation controls
// -----------------------------------------------------------------------------

AstralErr cpu_session_grammar_set_gbnf(void* session_ctx, AstralSpanU8 gbnf, AstralSpanU8 root) {
    if (session_ctx == nullptr || gbnf.data == nullptr || gbnf.len == 0) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->model == nullptr || session->model->vocab == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    // llama.cpp expects NUL-terminated strings.
    const size_t g_len = static_cast<size_t>(gbnf.len);
    char* g = static_cast<char*>(astral::core::runtime_alloc(g_len + 1, 1));
    if (g == nullptr) {
        return ASTRAL_E_NOMEM;
    }
    std::memcpy(g, gbnf.data, g_len);
    g[g_len] = '\0';

    const size_t r_len = static_cast<size_t>(root.len);
    char* r = nullptr;
    if (root.data != nullptr && root.len > 0) {
        r = static_cast<char*>(astral::core::runtime_alloc(r_len + 1, 1));
        if (r == nullptr) {
            astral::core::runtime_free(g, g_len + 1, 1);
            return ASTRAL_E_NOMEM;
        }
        std::memcpy(r, root.data, r_len);
        r[r_len] = '\0';
    }

    llama_sampler* new_grammars[kCpuMaxSlots]{};
    const char* root_name = r != nullptr ? r : "root";
    for (uint32_t i = 0; i < session->n_slots; ++i) {
        new_grammars[i] = llama_sampler_init_grammar(session->model->vocab, g, root_name);
        if (new_grammars[i] == nullptr) {
            for (uint32_t j = 0; j < i; ++j) {
                llama_sampler_free(new_grammars[j]);
            }
            astral::core::runtime_free(g, g_len + 1, 1);
            astral::core::runtime_free(r, r_len + 1, 1);
            return ASTRAL_E_INVALID;
        }
    }

    astral::core::runtime_free(g, g_len + 1, 1);
    astral::core::runtime_free(r, r_len + 1, 1);

    for (uint32_t i = 0; i < session->n_slots; ++i) {
        if (session->grammar[i] != nullptr) {
            llama_sampler_free(session->grammar[i]);
        }
        session->grammar[i] = new_grammars[i];
    }
    return ASTRAL_OK;
}

[[maybe_unused]] AstralErr cpu_session_grammar_set_json_schema(void* session_ctx, AstralSpanU8 json_schema) {
#if ASTRAL_ENABLE_JSON_SCHEMA_GRAMMAR
    if (session_ctx == nullptr || json_schema.data == nullptr || json_schema.len == 0) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->model == nullptr || session->model->vocab == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    const char* begin = reinterpret_cast<const char*>(json_schema.data);
    const char* end = begin + json_schema.len;
    std::string schema_text(begin, end);

    nlohmann::ordered_json schema = nlohmann::ordered_json::parse(schema_text, /*cb=*/nullptr, /*allow_exceptions=*/false);
    if (schema.is_discarded()) {
        return ASTRAL_E_INVALID;
    }

    std::string gbnf;
    try {
        gbnf = json_schema_to_grammar(schema, /*force_gbnf=*/true);
    } catch (...) {
        return ASTRAL_E_INVALID;
    }

    AstralSpanU8 g{};
    g.data = reinterpret_cast<const uint8_t*>(gbnf.data());
    g.len = static_cast<uint32_t>(gbnf.size());

    AstralSpanU8 root{};
    return cpu_session_grammar_set_gbnf(session_ctx, g, root);
#else
    (void)session_ctx;
    (void)json_schema;
    return ASTRAL_E_UNSUPPORTED;
#endif
}

AstralErr cpu_session_grammar_clear(void* session_ctx) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    for (uint32_t i = 0; i < session->n_slots; ++i) {
        if (session->grammar[i] != nullptr) {
            llama_sampler_free(session->grammar[i]);
            session->grammar[i] = nullptr;
        }
    }
    return ASTRAL_OK;
}

AstralErr cpu_session_apply_grammar(void* session_ctx, uint32_t* tokens, float* logits, uint32_t count) {
    if (session_ctx == nullptr || tokens == nullptr || logits == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (count == 0 || session->active_slot >= session->n_slots) {
        return ASTRAL_OK;
    }
    llama_sampler* smpl = session->grammar[session->active_slot];
    if (smpl == nullptr) {
        return ASTRAL_OK;
    }

    llama_token_data* data =
        static_cast<llama_token_data*>(ASTRAL_ALLOCA(static_cast<size_t>(count) * sizeof(llama_token_data)));

    for (uint32_t i = 0; i < count; ++i) {
        data[i].id = static_cast<llama_token>(tokens[i]);
        data[i].logit = logits[i];
        data[i].p = 0.0f;
    }

    llama_token_data_array arr{};
    arr.data = data;
    arr.size = count;
    arr.selected = -1;
    arr.sorted = false;

    llama_sampler_apply(smpl, &arr);

    const float neg_inf = -std::numeric_limits<float>::infinity();
    const size_t n = arr.size < static_cast<size_t>(count) ? arr.size : static_cast<size_t>(count);
    for (size_t i = 0; i < n; ++i) {
        tokens[i] = static_cast<uint32_t>(arr.data[i].id);
        logits[i] = arr.data[i].logit;
    }
    for (size_t i = n; i < count; ++i) {
        logits[i] = neg_inf;
    }

    return ASTRAL_OK;
}

AstralErr cpu_session_grammar_set_gbnf_for_slot(void* session_ctx, uint32_t slot_id, AstralSpanU8 gbnf, AstralSpanU8 root) {
    if (session_ctx == nullptr || gbnf.data == nullptr || gbnf.len == 0) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->model == nullptr || session->model->vocab == nullptr) {
        return ASTRAL_E_BACKEND;
    }
    if (slot_id >= session->n_slots) {
        return ASTRAL_E_INVALID;
    }

    // llama.cpp expects NUL-terminated strings.
    const size_t g_len = static_cast<size_t>(gbnf.len);
    char* g = static_cast<char*>(astral::core::runtime_alloc(g_len + 1, 1));
    if (g == nullptr) {
        return ASTRAL_E_NOMEM;
    }
    std::memcpy(g, gbnf.data, g_len);
    g[g_len] = '\0';

    const size_t r_len = static_cast<size_t>(root.len);
    char* r = nullptr;
    if (root.data != nullptr && root.len > 0) {
        r = static_cast<char*>(astral::core::runtime_alloc(r_len + 1, 1));
        if (r == nullptr) {
            astral::core::runtime_free(g, g_len + 1, 1);
            return ASTRAL_E_NOMEM;
        }
        std::memcpy(r, root.data, r_len);
        r[r_len] = '\0';
    }

    const char* root_name = r != nullptr ? r : "root";
    llama_sampler* sampler = llama_sampler_init_grammar(session->model->vocab, g, root_name);

    astral::core::runtime_free(g, g_len + 1, 1);
    astral::core::runtime_free(r, r_len + 1, 1);

    if (sampler == nullptr) {
        return ASTRAL_E_INVALID;
    }

    if (session->grammar[slot_id] != nullptr) {
        llama_sampler_free(session->grammar[slot_id]);
    }
    session->grammar[slot_id] = sampler;
    return ASTRAL_OK;
}

[[maybe_unused]] AstralErr cpu_session_grammar_set_json_schema_for_slot(void* session_ctx, uint32_t slot_id, AstralSpanU8 json_schema) {
#if ASTRAL_ENABLE_JSON_SCHEMA_GRAMMAR
    if (session_ctx == nullptr || json_schema.data == nullptr || json_schema.len == 0) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->model == nullptr || session->model->vocab == nullptr) {
        return ASTRAL_E_BACKEND;
    }
    if (slot_id >= session->n_slots) {
        return ASTRAL_E_INVALID;
    }

    const char* begin = reinterpret_cast<const char*>(json_schema.data);
    const char* end = begin + json_schema.len;
    std::string schema_text(begin, end);

    nlohmann::ordered_json schema = nlohmann::ordered_json::parse(schema_text, /*cb=*/nullptr, /*allow_exceptions=*/false);
    if (schema.is_discarded()) {
        return ASTRAL_E_INVALID;
    }

    std::string gbnf;
    try {
        gbnf = json_schema_to_grammar(schema, /*force_gbnf=*/true);
    } catch (...) {
        return ASTRAL_E_INVALID;
    }

    AstralSpanU8 g{};
    g.data = reinterpret_cast<const uint8_t*>(gbnf.data());
    g.len = static_cast<uint32_t>(gbnf.size());

    AstralSpanU8 root{};
    return cpu_session_grammar_set_gbnf_for_slot(session_ctx, slot_id, g, root);
#else
    (void)session_ctx;
    (void)slot_id;
    (void)json_schema;
    return ASTRAL_E_UNSUPPORTED;
#endif
}

AstralErr cpu_session_grammar_clear_for_slot(void* session_ctx, uint32_t slot_id) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (slot_id >= session->n_slots) {
        return ASTRAL_E_INVALID;
    }
    if (session->grammar[slot_id] != nullptr) {
        llama_sampler_free(session->grammar[slot_id]);
        session->grammar[slot_id] = nullptr;
    }
    return ASTRAL_OK;
}

AstralErr cpu_session_apply_grammar_for_slot(void* session_ctx, uint32_t slot_id, uint32_t* tokens, float* logits, uint32_t count) {
    if (session_ctx == nullptr || tokens == nullptr || logits == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (count == 0 || slot_id >= session->n_slots) {
        return ASTRAL_OK;
    }
    llama_sampler* smpl = session->grammar[slot_id];
    if (smpl == nullptr) {
        return ASTRAL_OK;
    }

    llama_token_data* data =
        static_cast<llama_token_data*>(ASTRAL_ALLOCA(static_cast<size_t>(count) * sizeof(llama_token_data)));

    for (uint32_t i = 0; i < count; ++i) {
        data[i].id = static_cast<llama_token>(tokens[i]);
        data[i].logit = logits[i];
        data[i].p = 0.0f;
    }

    llama_token_data_array arr{};
    arr.data = data;
    arr.size = count;
    arr.selected = -1;
    arr.sorted = false;

    llama_sampler_apply(smpl, &arr);

    const float neg_inf = -std::numeric_limits<float>::infinity();
    const size_t n = arr.size < static_cast<size_t>(count) ? arr.size : static_cast<size_t>(count);
    for (size_t i = 0; i < n; ++i) {
        tokens[i] = static_cast<uint32_t>(arr.data[i].id);
        logits[i] = arr.data[i].logit;
    }
    for (size_t i = n; i < count; ++i) {
        logits[i] = neg_inf;
    }

    return ASTRAL_OK;
}

AstralErr cpu_session_set_slot(void* session_ctx, uint32_t slot_id) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (slot_id >= session->n_slots) {
        return ASTRAL_E_INVALID;
    }
    session->active_slot = slot_id;
    return ASTRAL_OK;
}

AstralErr cpu_session_slot_pos(void* session_ctx, uint32_t slot_id, uint32_t* out_pos) {
    if (session_ctx == nullptr || out_pos == nullptr) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->ctx == nullptr || session->model == nullptr) {
        return ASTRAL_E_BACKEND;
    }
    if (slot_id >= session->n_slots) {
        return ASTRAL_E_INVALID;
    }
    const int32_t pos = session->slot_pos[slot_id];
    *out_pos = pos > 0 ? static_cast<uint32_t>(pos) : 0u;
    return ASTRAL_OK;
}

AstralErr cpu_session_state_size(void* session_ctx, uint64_t* out_bytes) {
    if (session_ctx == nullptr || out_bytes == nullptr) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->ctx == nullptr) {
        return ASTRAL_E_BACKEND;
    }
    *out_bytes = static_cast<uint64_t>(sizeof(CpuStateHeaderV1)) +
                 static_cast<uint64_t>(llama_state_get_size(session->ctx));
    return ASTRAL_OK;
}

AstralErr cpu_session_state_save(void* session_ctx, uint8_t* out_bytes, uint64_t capacity, uint64_t* out_written) {
    if (session_ctx == nullptr || out_bytes == nullptr || out_written == nullptr) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->ctx == nullptr) {
        return ASTRAL_E_BACKEND;
    }
    const size_t llama_bytes = llama_state_get_size(session->ctx);
    const size_t need = sizeof(CpuStateHeaderV1) + llama_bytes;
    if (capacity < need) {
        return ASTRAL_E_NOMEM;
    }

    CpuStateHeaderV1 h{};
    h.magic = kCpuStateMagic;
    h.version = kCpuStateVersion;
    h.header_bytes = static_cast<uint16_t>(sizeof(CpuStateHeaderV1));
    h.llama_bytes = static_cast<uint64_t>(llama_bytes);
    h.active_slot = session->active_slot;
    h.n_slots = session->n_slots;
    for (uint32_t i = 0; i < kCpuMaxSlots; ++i) {
        h.slot_last_token[i] = session->slot_last_token[i];
        h.slot_last_token_valid[i] = session->slot_last_token_valid[i] ? 1u : 0u;
    }
    std::memcpy(out_bytes, &h, sizeof(h));

    const size_t wrote = llama_state_get_data(session->ctx, out_bytes + sizeof(CpuStateHeaderV1), llama_bytes);
    if (wrote == 0) {
        return ASTRAL_E_BACKEND;
    }
    *out_written = static_cast<uint64_t>(sizeof(CpuStateHeaderV1) + wrote);
    return ASTRAL_OK;
}

AstralErr cpu_session_state_load(void* session_ctx, const uint8_t* bytes, uint64_t size) {
    if (session_ctx == nullptr || bytes == nullptr || size == 0) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->ctx == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    CpuStateHeaderV1 h{};
    const uint8_t* llama_bytes = bytes;
    uint64_t llama_size = size;
    bool has_cpu_header = false;
    if (size >= sizeof(CpuStateHeaderV1)) {
        std::memcpy(&h, bytes, sizeof(h));
        if (h.magic == kCpuStateMagic && h.version == kCpuStateVersion &&
            h.header_bytes == sizeof(CpuStateHeaderV1) &&
            h.llama_bytes <= size - static_cast<uint64_t>(sizeof(CpuStateHeaderV1))) {
            has_cpu_header = true;
            llama_bytes = bytes + sizeof(CpuStateHeaderV1);
            llama_size = h.llama_bytes;
        }
    }

    const size_t read = llama_state_set_data(session->ctx, llama_bytes, static_cast<size_t>(llama_size));
    if (read != static_cast<size_t>(llama_size)) {
        return ASTRAL_E_INVALID;
    }

    if (has_cpu_header) {
        session->active_slot = h.active_slot < session->n_slots ? h.active_slot : 0u;
        const uint32_t n = h.n_slots < kCpuMaxSlots ? h.n_slots : kCpuMaxSlots;
        for (uint32_t i = 0; i < n; ++i) {
            session->slot_last_token[i] = h.slot_last_token[i];
            session->slot_last_token_valid[i] = h.slot_last_token_valid[i] != 0;
        }
        for (uint32_t i = n; i < kCpuMaxSlots; ++i) {
            session->slot_last_token[i] = 0;
            session->slot_last_token_valid[i] = false;
        }
    } else {
        for (uint32_t i = 0; i < kCpuMaxSlots; ++i) {
            session->slot_last_token[i] = 0;
            session->slot_last_token_valid[i] = false;
        }
    }

    // Reconstruct per-slot token positions from the KV cache.
    llama_memory_t mem = llama_get_memory(session->ctx);
    if (mem != nullptr) {
        for (uint32_t i = 0; i < session->n_slots; ++i) {
            const llama_pos pmax = llama_memory_seq_pos_max(mem, static_cast<llama_seq_id>(i));
            session->slot_pos[i] = pmax >= 0 ? (static_cast<int32_t>(pmax) + 1) : 0;
            session->slot_has_logits[i] = false;
        }
    }

    if (has_cpu_header && session->active_slot < session->n_slots &&
        session->slot_last_token_valid[session->active_slot] &&
        session->slot_pos[session->active_slot] > 0) {
        const uint32_t slot = session->active_slot;
        const int32_t pos = session->slot_pos[slot] - 1;
        if (mem == nullptr || !llama_memory_seq_rm(mem, static_cast<llama_seq_id>(slot), pos, pos + 1)) {
            return ASTRAL_E_BACKEND;
        }

        llama_token t = static_cast<llama_token>(session->slot_last_token[slot]);
        llama_batch batch{};
        batch.n_tokens = 1;
        batch.token = &t;
        batch.embd = nullptr;
        batch.pos = reinterpret_cast<llama_pos*>(session->batch_pos);
        batch.n_seq_id = session->batch_n_seq_id;
        batch.seq_id = reinterpret_cast<llama_seq_id**>(session->batch_seq_id_ptrs);
        batch.logits = session->batch_logits;

        session->batch_pos[0] = pos;
        session->batch_seq_id_storage[0] = static_cast<int32_t>(slot);
        session->batch_logits[0] = 1;

        ASTRAL_ZONE_N("astral.cpu.rebuild_logits_after_state_load");
        const int32_t rc = llama_decode(session->ctx, batch);
        if (rc != 0) {
            return ASTRAL_E_BACKEND;
        }
        session->slot_pos[slot] = pos + 1;
        session->slot_has_logits[slot] = true;
    }

    // Grammar state is not serialized in llama_state_*; reset samplers.
    for (uint32_t i = 0; i < session->n_slots; ++i) {
        if (session->grammar[i] != nullptr) {
            llama_sampler_reset(session->grammar[i]);
        }
    }
    return ASTRAL_OK;
}

void* cpu_model_adapter_load(void* model_ctx, AstralSpanU8 path, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }
    if (model_ctx == nullptr || path.data == nullptr || path.len == 0) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);
    const size_t path_len = static_cast<size_t>(path.len);
    char* p = static_cast<char*>(astral::core::runtime_alloc(path_len + 1, 1));
    if (p == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }
    std::memcpy(p, path.data, path_len);
    p[path_len] = '\0';

    llama_adapter_lora* a = llama_adapter_lora_init(model->model, p);
    astral::core::runtime_free(p, path_len + 1, 1);

    if (a == nullptr) {
        *out_err = ASTRAL_E_BACKEND;
        return nullptr;
    }
    *out_err = ASTRAL_OK;
    return a;
}

void cpu_model_adapter_unload(void* model_ctx, void* adapter_ctx) {
    (void)model_ctx;
    if (adapter_ctx == nullptr) {
        return;
    }
    llama_adapter_lora_free(static_cast<llama_adapter_lora*>(adapter_ctx));
}

AstralErr cpu_session_adapter_clear(void* session_ctx) {
    if (session_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->ctx == nullptr) {
        return ASTRAL_E_BACKEND;
    }
    const int32_t rc = llama_set_adapters_lora(session->ctx, nullptr, 0, nullptr);
    if (rc != 0) {
        return ASTRAL_E_BACKEND;
    }
    session->adapter_count = 0;
    for (uint32_t i = 0; i < ASTRAL_SESSION_ADAPTERS_MAX; ++i) {
        session->adapters[i] = nullptr;
        session->adapter_scales[i] = 0.0f;
    }
    return ASTRAL_OK;
}

AstralErr cpu_session_adapter_add(void* session_ctx, void* adapter_ctx, float scale) {
    if (session_ctx == nullptr || adapter_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }
    CpuSession* session = static_cast<CpuSession*>(session_ctx);
    if (session->ctx == nullptr) {
        return ASTRAL_E_BACKEND;
    }
    if (session->adapter_count >= ASTRAL_SESSION_ADAPTERS_MAX) {
        return ASTRAL_E_NOMEM;
    }

    const uint32_t index = session->adapter_count;
    session->adapters[index] = static_cast<llama_adapter_lora*>(adapter_ctx);
    session->adapter_scales[index] = scale;
    const uint32_t next_count = index + 1u;
    const int32_t rc = llama_set_adapters_lora(session->ctx, session->adapters, next_count, session->adapter_scales);
    if (rc != 0) {
        session->adapters[index] = nullptr;
        session->adapter_scales[index] = 0.0f;
        return ASTRAL_E_BACKEND;
    }
    session->adapter_count = next_count;
    return ASTRAL_OK;
}

void* cpu_embedder_create(void* model_ctx, AstralErr* out_err) {
    ASTRAL_ZONE_N("astral.cpu.embedder_create");

    if (out_err == nullptr) {
        return nullptr;
    }
    if (model_ctx == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    CpuModel* model = static_cast<CpuModel*>(model_ctx);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = model->n_ctx;
    ctx_params.n_batch = model->n_batch;
    ctx_params.n_ubatch = ctx_params.n_batch;

    const uint32_t threads_default = default_threads_fallback();
    ctx_params.n_threads = static_cast<int32_t>(clamp_threads(model->n_threads, threads_default));
    ctx_params.n_threads_batch =
        static_cast<int32_t>(clamp_threads(model->n_threads_batch, static_cast<uint32_t>(ctx_params.n_threads)));
    ctx_params.n_seq_max = 1;

    ctx_params.embeddings = true;
    ctx_params.no_perf = true;
    // NOTE: llama.cpp clears pooled-embedding caches using a hash-map each call, which
    // can allocate. Keep pooling off and do any pooling in Astral without heap use.
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    if (llama_model_has_encoder(model->model)) {
        ctx_params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    } else {
        ctx_params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    llama_context* ctx = llama_init_from_model(model->model, ctx_params);
    if (ctx == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    llama_set_embeddings(ctx, true);

    CpuEmbedder* emb = astral::core::runtime_new<CpuEmbedder>();
    if (emb == nullptr) {
        llama_free(ctx);
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }

    emb->model = model;
    emb->ctx = ctx;
    emb->use_encode = llama_model_has_encoder(model->model) ? 1 : 0;
    emb->mean_pooling = emb->use_encode;
    *out_err = ASTRAL_OK;
    return emb;
}

void cpu_embedder_destroy(void* embedder_ctx) {
    ASTRAL_ZONE_N("astral.cpu.embedder_destroy");

    if (embedder_ctx == nullptr) {
        return;
    }

    CpuEmbedder* emb = static_cast<CpuEmbedder*>(embedder_ctx);
    if (emb->ctx) {
        llama_free(emb->ctx);
    }
    astral::core::runtime_delete(emb);
}

AstralErr cpu_embedder_reset(void* embedder_ctx) {
    ASTRAL_ZONE_N("astral.cpu.embedder_reset");

    if (embedder_ctx == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuEmbedder* emb = static_cast<CpuEmbedder*>(embedder_ctx);
    if (emb->ctx == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    llama_memory_t mem = llama_get_memory(emb->ctx);
    if (mem == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    llama_memory_clear(mem, /*data=*/true);
    return ASTRAL_OK;
}

AstralErr cpu_embedder_embed(void* embedder_ctx, const int32_t* tokens, uint32_t count, float* out_vec, uint32_t vec_dim) {
    ASTRAL_ZONE_N("astral.cpu.embedder_embed_text");

    if (embedder_ctx == nullptr || out_vec == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuEmbedder* emb = static_cast<CpuEmbedder*>(embedder_ctx);
    if (emb->ctx == nullptr || emb->model == nullptr || emb->model->model == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    if (tokens == nullptr && count > 0) {
        return ASTRAL_E_INVALID;
    }

    if (vec_dim < emb->model->n_embd) {
        return ASTRAL_E_NOMEM;
    }

    // Ensure a clean KV/memory for this request.
    (void)cpu_embedder_reset(embedder_ctx);

    if (count == 0) {
        return ASTRAL_E_INVALID;
    }

    const uint32_t dim = emb->model->n_embd;
    const bool use_encode = emb->use_encode != 0;
    const bool mean_pooling = emb->mean_pooling != 0;

    if (mean_pooling) {
        std::memset(out_vec, 0, static_cast<size_t>(dim) * sizeof(float));
    }

    const uint32_t batch_size = emb->model->n_batch > 0 ? emb->model->n_batch : count;
    uint32_t total = 0;
    for (uint32_t i = 0; i < count; i += batch_size) {
        const uint32_t n = (count - i) < batch_size ? (count - i) : batch_size;
        llama_token* chunk = const_cast<llama_token*>(reinterpret_cast<const llama_token*>(tokens + i));
        const llama_batch batch = llama_batch_get_one(chunk, static_cast<int32_t>(n));

        const int32_t rc = use_encode ? llama_encode(emb->ctx, batch) : llama_decode(emb->ctx, batch);
        if (rc != 0) {
            return ASTRAL_E_BACKEND;
        }

        float* base = llama_get_embeddings(emb->ctx);
        if (base == nullptr) {
            return ASTRAL_E_BACKEND;
        }

        if (mean_pooling) {
            for (uint32_t t = 0; t < n; ++t) {
                const float* row = base + static_cast<size_t>(t) * static_cast<size_t>(dim);
                for (uint32_t k = 0; k < dim; ++k) {
                    out_vec[k] += row[k];
                }
            }
            total += n;
        } else {
            const float* last = base + static_cast<size_t>(n - 1u) * static_cast<size_t>(dim);
            std::memcpy(out_vec, last, static_cast<size_t>(dim) * sizeof(float));
        }
    }

    if (mean_pooling) {
        if (total == 0) {
            return ASTRAL_E_BACKEND;
        }
        const float inv = 1.0f / static_cast<float>(total);
        for (uint32_t k = 0; k < dim; ++k) {
            out_vec[k] *= inv;
        }
    }

    return ASTRAL_OK;
}

AstralErr cpu_embedder_embed_multimodal(void* embedder_ctx,
                                        AstralSpanU8 text,
                                        const AstralImageDesc* image,
                                        const AstralAudioDesc* audio,
                                        float* out_vec,
                                        uint32_t vec_dim);

AstralErr cpu_embedder_embed_image(void* embedder_ctx, const AstralImageDesc* image, float* out_vec, uint32_t vec_dim) {
    ASTRAL_ZONE_N("astral.cpu.embedder_embed_image");

#if !ASTRAL_ENABLE_MTMD
    (void)embedder_ctx;
    (void)image;
    (void)out_vec;
    (void)vec_dim;
    return ASTRAL_E_UNSUPPORTED;
#else
    return cpu_embedder_embed_multimodal(embedder_ctx, AstralSpanU8{}, image, nullptr, out_vec, vec_dim);
#endif
}

AstralErr cpu_embedder_embed_audio(void* embedder_ctx, const AstralAudioDesc* audio, float* out_vec, uint32_t vec_dim) {
    ASTRAL_ZONE_N("astral.cpu.embedder_embed_audio");

#if !ASTRAL_ENABLE_MTMD
    (void)embedder_ctx;
    (void)audio;
    (void)out_vec;
    (void)vec_dim;
    return ASTRAL_E_UNSUPPORTED;
#else
    return cpu_embedder_embed_multimodal(embedder_ctx, AstralSpanU8{}, nullptr, audio, out_vec, vec_dim);
#endif
}

AstralErr cpu_embedder_embed_multimodal(void* embedder_ctx,
                                        AstralSpanU8 text,
                                        const AstralImageDesc* image,
                                        const AstralAudioDesc* audio,
                                        float* out_vec,
                                        uint32_t vec_dim) {
    ASTRAL_ZONE_N("astral.cpu.embedder_embed_multimodal");

#if !ASTRAL_ENABLE_MTMD
    (void)embedder_ctx;
    (void)text;
    (void)image;
    (void)audio;
    (void)out_vec;
    (void)vec_dim;
    return ASTRAL_E_UNSUPPORTED;
#else
    if (embedder_ctx == nullptr || out_vec == nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuEmbedder* emb = static_cast<CpuEmbedder*>(embedder_ctx);
    if (emb->ctx == nullptr || emb->model == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    if (image != nullptr && audio != nullptr) {
        return ASTRAL_E_INVALID;
    }

    CpuModel* model = emb->model;
    if (vec_dim < model->n_embd) {
        return ASTRAL_E_NOMEM;
    }

    if (image == nullptr && audio == nullptr) {
        return ASTRAL_E_INVALID;
    }

    if (model->mtmd == nullptr) {
        return ASTRAL_E_UNSUPPORTED;
    }

    (void)cpu_embedder_reset(embedder_ctx);

    const char* marker = mtmd_default_marker();
    const size_t marker_len = std::strlen(marker);
    size_t text_len = text.data ? static_cast<size_t>(text.len) : 0;
    const bool need_space = text_len > 0 && (text.data[text_len - 1] != ' ');
    const size_t prompt_len = text_len + (need_space ? 1 : 0) + marker_len;

    char* prompt = static_cast<char*>(astral::core::runtime_alloc(prompt_len + 1, 1));
    if (prompt == nullptr) {
        return ASTRAL_E_NOMEM;
    }
    size_t off = 0;
    if (text_len > 0) {
        std::memcpy(prompt, text.data, text_len);
        off += text_len;
    }
    if (need_space) {
        prompt[off++] = ' ';
    }
    std::memcpy(prompt + off, marker, marker_len);
    off += marker_len;
    prompt[off] = '\0';

    MediaBuffer buf{};
    AstralErr prep = ASTRAL_E_UNSUPPORTED;
    mtmd_bitmap* bmp = nullptr;
    if (image != nullptr) {
        prep = prepare_image_rgb8(image, &buf);
        if (prep == ASTRAL_OK) {
            bmp = mtmd_bitmap_init(image->width, image->height, buf.data);
        }
    } else if (audio != nullptr) {
        prep = prepare_audio_f32(audio, &buf, model->audio_sample_rate);
        if (prep == ASTRAL_OK) {
            const size_t sample_count = buf.size / sizeof(float);
            bmp = mtmd_bitmap_init_from_audio(sample_count, reinterpret_cast<const float*>(buf.data));
        }
    }
    media_buffer_free(&buf);

    if (prep != ASTRAL_OK) {
        astral::core::runtime_free(prompt, prompt_len + 1, 1);
        return prep;
    }
    if (bmp == nullptr) {
        astral::core::runtime_free(prompt, prompt_len + 1, 1);
        return ASTRAL_E_NOMEM;
    }

    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    if (chunks == nullptr) {
        mtmd_bitmap_free(bmp);
        astral::core::runtime_free(prompt, prompt_len + 1, 1);
        return ASTRAL_E_NOMEM;
    }

    mtmd_input_text inp{};
    inp.text = prompt;
    inp.add_special = true;
    inp.parse_special = false;
    const mtmd_bitmap* bitmaps[1] = { bmp };

    int32_t rc = 0;
    mtmd_lock_acquire(model->mtmd_lock);
    rc = mtmd_tokenize(model->mtmd, chunks, &inp, bitmaps, 1);
    if (rc == 0) {
        llama_pos new_n_past = 0;
        rc = mtmd_helper_eval_chunks(
            model->mtmd,
            emb->ctx,
            chunks,
            0,
            0,
            static_cast<int32_t>(model->n_batch),
            true,
            &new_n_past
        );
    }
    mtmd_lock_release(model->mtmd_lock);

    // Capture last-chunk token count before freeing chunks.
    int32_t last_tokens = 0;
    if (rc == 0) {
        const size_t n_chunks = mtmd_input_chunks_size(chunks);
        if (n_chunks > 0) {
            const mtmd_input_chunk* last = mtmd_input_chunks_get(chunks, n_chunks - 1);
            last_tokens = static_cast<int32_t>(mtmd_input_chunk_get_n_tokens(last));
        }
    }

    astral::core::runtime_free(prompt, prompt_len + 1, 1);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);

    if (rc != 0) {
        return ASTRAL_E_BACKEND;
    }

    if (last_tokens <= 0) {
        return ASTRAL_E_BACKEND;
    }

    // Grab last-token embedding from the final batch.
    float* base = llama_get_embeddings(emb->ctx);
    if (base == nullptr) {
        return ASTRAL_E_BACKEND;
    }

    const int32_t batch = static_cast<int32_t>(model->n_batch > 0 ? model->n_batch : static_cast<uint32_t>(last_tokens));
    int32_t last_batch = last_tokens % batch;
    if (last_batch == 0) {
        last_batch = batch;
    }
    const float* last_row = base + static_cast<size_t>(last_batch - 1) * static_cast<size_t>(model->n_embd);
    std::memcpy(out_vec, last_row, static_cast<size_t>(model->n_embd) * sizeof(float));
    return ASTRAL_OK;
#endif
}

static const BackendOps kCpuBackendOps = [] {
  BackendOps ops{};
  ops.model_load = cpu_model_load;
  ops.model_unload = cpu_model_unload;
  ops.tokenize = cpu_tokenize;
  ops.detokenize = cpu_detokenize;
  ops.model_info = cpu_model_info;
  ops.model_special_tokens = cpu_model_special_tokens;
  ops.model_embedding_dim = cpu_model_embedding_dim;
  ops.model_media_init = cpu_model_media_init;
  ops.model_media_info = cpu_model_media_info;
  ops.session_create = cpu_session_create;
  ops.session_create_ex = cpu_session_create_ex;
  ops.session_destroy = cpu_session_destroy;
  ops.session_reset = cpu_session_reset;
  ops.session_feed = cpu_session_feed;
  ops.session_feed_image = cpu_session_feed_image;
  ops.session_feed_audio = cpu_session_feed_audio;
  ops.session_logits = cpu_session_logits;
  ops.session_accept = cpu_session_accept;
  ops.session_batch_eval = cpu_session_batch_eval;
  ops.session_batch_logits = cpu_session_batch_logits;
  ops.session_slot_reset = cpu_session_slot_reset;
  ops.embedder_create = cpu_embedder_create;
  ops.embedder_destroy = cpu_embedder_destroy;
  ops.embedder_reset = cpu_embedder_reset;
  ops.embedder_embed = cpu_embedder_embed;
  ops.embedder_embed_image = cpu_embedder_embed_image;
  ops.embedder_embed_audio = cpu_embedder_embed_audio;
  ops.embedder_embed_multimodal = cpu_embedder_embed_multimodal;
  ops.session_grammar_set_gbnf = cpu_session_grammar_set_gbnf;
  ops.session_grammar_set_json_schema =
      ASTRAL_ENABLE_JSON_SCHEMA_GRAMMAR ? cpu_session_grammar_set_json_schema : nullptr;
  ops.session_grammar_clear = cpu_session_grammar_clear;
  ops.session_apply_grammar = cpu_session_apply_grammar;
  ops.session_grammar_set_gbnf_for_slot = cpu_session_grammar_set_gbnf_for_slot;
  ops.session_grammar_set_json_schema_for_slot =
      ASTRAL_ENABLE_JSON_SCHEMA_GRAMMAR ? cpu_session_grammar_set_json_schema_for_slot : nullptr;
  ops.session_grammar_clear_for_slot = cpu_session_grammar_clear_for_slot;
  ops.session_apply_grammar_for_slot = cpu_session_apply_grammar_for_slot;
  ops.session_state_size = cpu_session_state_size;
  ops.session_state_save = cpu_session_state_save;
  ops.session_state_load = cpu_session_state_load;
  ops.model_adapter_load = cpu_model_adapter_load;
  ops.model_adapter_unload = cpu_model_adapter_unload;
  ops.session_adapter_clear = cpu_session_adapter_clear;
  ops.session_adapter_add = cpu_session_adapter_add;
  ops.session_set_slot = cpu_session_set_slot;
  ops.session_slot_pos = cpu_session_slot_pos;
  return ops;
}();

static const BackendProvider kCpuBackendProvider = {
    /*name=*/"cpu",
    /*ops=*/&kCpuBackendOps,
    /*supports_gpu=*/false,
    /*min_gpu_layers=*/0,
};

} // namespace

const BackendProvider* builtin_cpu_backend_provider() {
    return &kCpuBackendProvider;
}

} // namespace astral::backend
