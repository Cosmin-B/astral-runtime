#include "astral_rt.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace {

static uint64_t now_ms() {
#if defined(__linux__) || defined(__APPLE__)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000ULL + static_cast<uint64_t>(ts.tv_nsec) / 1000000ULL;
#else
    return static_cast<uint64_t>(time(nullptr)) * 1000ULL;
#endif
}

static bool arg_eq(const char* a, const char* b) {
    if (a == nullptr || b == nullptr) {
        return false;
    }
    return std::strcmp(a, b) == 0;
}

static bool parse_u32(const char* s, uint32_t* out) {
    if (s == nullptr || out == nullptr) {
        return false;
    }
    char* end = nullptr;
    unsigned long v = std::strtoul(s, &end, 10);
    if (end == s) {
        return false;
    }
    *out = static_cast<uint32_t>(v);
    return true;
}

static bool parse_u64(const char* s, uint64_t* out) {
    if (s == nullptr || out == nullptr) {
        return false;
    }
    char* end = nullptr;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s) {
        return false;
    }
    *out = static_cast<uint64_t>(v);
    return true;
}

static void usage(const char* exe) {
    std::fprintf(stderr,
                 "Usage: %s --model <path.gguf> [options]\n"
                 "\n"
                 "Options:\n"
                 "  --backend <name>        Backend override (default: cpu)\n"
                 "  --prompt <text>         Prompt text (default: \"hi\")\n"
                 "  --tokens <N>            Max tokens (default: 128)\n"
                 "  --sink <stdout|none>    Output sink (default: stdout)\n"
                 "  --reserve-mb <N>        Reserve virtual memory in MiB (default: 256)\n"
                 "  --threads <N>           Runtime thread count (default: 2)\n"
                 "  --cancel-after-ms <N>   Request cancel after N ms (default: 0=disabled)\n"
                 "  --reset                 Run twice using session reset\n",
                 exe ? exe : "astral_embedded_cli");
}

} // namespace

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* backend_name = "cpu";
    const char* prompt = "hi";
    const char* mock_model = "infinite";

    uint32_t max_tokens = 128;
    uint64_t reserve_mb = 256;
    uint32_t runtime_threads = 2;
    uint32_t cancel_after_ms = 0;
    bool do_reset = false;
    bool sink_stdout = true;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (arg_eq(a, "--help") || arg_eq(a, "-h")) {
            usage(argv[0]);
            return 0;
        }
        if (arg_eq(a, "--model") && i + 1 < argc) {
            model_path = argv[++i];
            continue;
        }
        if (arg_eq(a, "--backend") && i + 1 < argc) {
            backend_name = argv[++i];
            continue;
        }
        if (arg_eq(a, "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
            continue;
        }
        if (arg_eq(a, "--tokens") && i + 1 < argc) {
            (void)parse_u32(argv[++i], &max_tokens);
            continue;
        }
        if (arg_eq(a, "--sink") && i + 1 < argc) {
            const char* v = argv[++i];
            if (v != nullptr && std::strcmp(v, "none") == 0) {
                sink_stdout = false;
            } else {
                sink_stdout = true;
            }
            continue;
        }
        if (arg_eq(a, "--reserve-mb") && i + 1 < argc) {
            (void)parse_u64(argv[++i], &reserve_mb);
            continue;
        }
        if (arg_eq(a, "--threads") && i + 1 < argc) {
            (void)parse_u32(argv[++i], &runtime_threads);
            continue;
        }
        if (arg_eq(a, "--cancel-after-ms") && i + 1 < argc) {
            (void)parse_u32(argv[++i], &cancel_after_ms);
            continue;
        }
        if (arg_eq(a, "--reset")) {
            do_reset = true;
            continue;
        }

        std::fprintf(stderr, "Unknown argument: %s\n", a ? a : "(null)");
        usage(argv[0]);
        return 2;
    }

    if (model_path == nullptr || model_path[0] == '\0') {
        if (backend_name != nullptr && std::strcmp(backend_name, "mock") == 0) {
            model_path = mock_model;
        } else {
            std::fprintf(stderr, "Missing required --model <path.gguf>\n");
            usage(argv[0]);
            return 2;
        }
    }

    AstralInit cfg{};
    cfg.reserve_bytes = reserve_mb * 1024ULL * 1024ULL;
    cfg.thread_count = runtime_threads;
    cfg.numa_node = 0xFFFFFFFFu;
    cfg.enable_hugepages = 0;
    AstralErr err = astral_init(&cfg);
    if (err == ASTRAL_E_UNSUPPORTED) {
        // Embedded-friendly fallback: use an Astral-owned arena instead of platform VM.
        AstralInit2 cfg2{};
        cfg2.base = cfg;
        cfg2.memory_mode = ASTRAL_MEMMODE_ARENA_OWNED;
        cfg2.arena.size = cfg.reserve_bytes;
        cfg2.arena.session_block_size = 2u * 1024u * 1024u;
        cfg2.arena.session_block_count = 0; // auto
        err = astral_init2(&cfg2);
    }
    if (err != ASTRAL_OK) {
        std::fprintf(stderr, "astral_init failed: %s (%s)\n", astral_error_string(err), astral_last_error());
        return 1;
    }

    AstralModelDesc model_desc{};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    model_desc.model_path.data = reinterpret_cast<const uint8_t*>(model_path);
    model_desc.model_path.len = static_cast<uint32_t>(std::strlen(model_path));
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend_name);
    model_desc.backend_name.len = static_cast<uint32_t>(std::strlen(backend_name));
    model_desc.gpu_layers = 0;
    model_desc.n_ctx = 0;
    model_desc.n_batch = 0;
    model_desc.n_threads = 0;
    model_desc.embeddings_only = 0;

    AstralHandle model = 0;
    err = astral_model_load(&model_desc, &model);
    if (err != ASTRAL_OK) {
        std::fprintf(stderr, "astral_model_load failed: %s (%s)\n", astral_error_string(err), astral_last_error());
        astral_shutdown();
        return 1;
    }

    AstralSessionDesc session_desc{};
    session_desc.model = model;
    session_desc.max_tokens = max_tokens;
    session_desc.temperature = 0.0f;
    session_desc.top_k = 0;
    session_desc.top_p = 1.0f;
    session_desc.stream_enabled = 1;
    session_desc.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&session_desc, &session);
    if (err != ASTRAL_OK) {
        std::fprintf(stderr, "astral_session_create failed: %s (%s)\n", astral_error_string(err), astral_last_error());
        astral_model_release(model);
        astral_shutdown();
        return 1;
    }

    const auto run_once = [&](const char* label) -> bool {
        std::fprintf(stderr, "[%s] prompt=%s\n", label ? label : "run", prompt);

        AstralSpanU8 chunk{};
        chunk.data = reinterpret_cast<const uint8_t*>(prompt);
        chunk.len = static_cast<uint32_t>(std::strlen(prompt));

        AstralErr e = astral_session_feed(session, chunk, 1);
        if (e != ASTRAL_OK) {
            std::fprintf(stderr, "[%s] session_feed failed: %s (%s)\n", label, astral_error_string(e), astral_last_error());
            return false;
        }

        e = astral_session_decode(session);
        if (e != ASTRAL_OK) {
            std::fprintf(stderr, "[%s] session_decode failed: %s (%s)\n", label, astral_error_string(e), astral_last_error());
            return false;
        }

        const uint64_t t0 = now_ms();
        bool cancel_sent = false;

        uint8_t buf[512];
        for (;;) {
            if (!cancel_sent && cancel_after_ms > 0) {
                const uint64_t dt = now_ms() - t0;
                if (dt >= static_cast<uint64_t>(cancel_after_ms)) {
                    (void)astral_session_cancel(session);
                    cancel_sent = true;
                }
            }

            AstralMutSpanU8 out{};
            out.data = buf;
            out.len = sizeof(buf);
            const int32_t n = astral_stream_read(session, out, 10);
            if (n == ASTRAL_E_TIMEOUT) {
                continue;
            }
            if (n < 0) {
                std::fprintf(stderr, "[%s] stream_read failed: %d (%s)\n", label, n, astral_last_error());
                return false;
            }
            if (n == 0) {
                break;
            }
            if (sink_stdout) {
                (void)std::fwrite(buf, 1, static_cast<size_t>(n), stdout);
                (void)std::fflush(stdout);
            }
        }

        e = astral_session_wait(session, 60000);
        if (e != ASTRAL_OK) {
            std::fprintf(stderr, "[%s] session_wait failed: %s (%s)\n", label, astral_error_string(e), astral_last_error());
            return false;
        }

        AstralStats stats{};
        e = astral_session_stats(session, &stats);
        if (e == ASTRAL_OK) {
            std::fprintf(stderr,
                         "\n[%s] tok/s=%.2f ttft_ms=%.2f committed=%llu reserved=%llu\n",
                         label,
                         stats.tok_per_s,
                         stats.t_first_token_ms,
                         static_cast<unsigned long long>(stats.bytes_committed),
                         static_cast<unsigned long long>(stats.bytes_reserved));
        }

        return true;
    };

    bool ok = run_once("run1");

    if (ok && do_reset) {
        err = astral_session_reset(session, &session_desc);
        if (err != ASTRAL_OK) {
            std::fprintf(stderr, "astral_session_reset failed: %s (%s)\n", astral_error_string(err), astral_last_error());
            ok = false;
        } else {
            ok = run_once("run2");
        }
    }

    astral_session_destroy(session);
    astral_model_release(model);
    astral_shutdown();
    return ok ? 0 : 1;
}
