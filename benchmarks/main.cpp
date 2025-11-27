#include "bench_clock.hpp"
#include "bench_common.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace astral::bench {
BenchResult bench_spsc_ring(uint64_t items);
BenchResult bench_mpsc_ring(uint32_t producers, uint64_t items_per_producer);
BenchResult bench_mpmc_queue(uint32_t producers, uint32_t consumers, uint64_t items_per_producer);
BenchResult bench_runtime_alloc_free(uint64_t iters, uint32_t size, bool arena_mode);
BenchResult bench_session_create_destroy(uint64_t iters, bool arena_mode);
BenchResult bench_model_load_release(uint64_t iters, bool arena_mode);
void bench_inference_print(uint32_t warmup_tokens, uint32_t measure_tokens);
void bench_embeddings_print(uint32_t dim_override, uint64_t iters);
void bench_feature_surfaces_print(void);
void bench_platform_print(uint64_t iters);
} // namespace astral::bench

namespace {

struct Options {
    uint64_t spsc_items = 10'000'000ull;
    bool run_mpsc = true;
    uint32_t mpsc_producers = 4;
    uint64_t mpsc_items_per_producer = 1'000'000ull;
    uint32_t mpmc_producers = 4;
    uint32_t mpmc_consumers = 4;
    uint64_t mpmc_items_per_producer = 1'000'000ull;
    bool run_spsc = true;
    bool run_mpmc = true;
    bool run_alloc = false;
    uint64_t alloc_iters = 5'000'000ull;
    uint32_t alloc_size = 64;
    bool run_lifecycle = false;
    uint64_t lifecycle_iters = 50'000ull;
    bool run_infer = false;
    uint32_t infer_warmup_tokens = 16;
    uint32_t infer_tokens = 128;
    bool run_embed = false;
    uint32_t embed_dim = 0;
    uint64_t embed_iters = 10000;
    bool run_features = false;
    bool run_platform = false;
    uint64_t platform_iters = 1000000;
};

bool parse_u64(const char* s, uint64_t* out) {
    if (s == nullptr || out == nullptr) {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (errno != 0 || end == s || (end && *end != '\0')) {
        return false;
    }
    *out = static_cast<uint64_t>(v);
    return true;
}

bool parse_u32(const char* s, uint32_t* out) {
    uint64_t v = 0;
    if (!parse_u64(s, &v) || v > 0xFFFFFFFFull) {
        return false;
    }
    *out = static_cast<uint32_t>(v);
    return true;
}

void print_usage(const char* argv0) {
    std::printf("Usage: %s [options]\n", argv0 ? argv0 : "astral_benchmarks");
    std::printf("Options:\n");
    std::printf("  --only <spsc|mpsc|mpmc|alloc|lifecycle|infer|embed|features|platform> Run only one benchmark\n");
    std::printf("  --alloc                       Run runtime_alloc/free microbench (vm + arena)\n");
    std::printf("  --alloc-iters <N>             Iterations for alloc bench (default: 5000000)\n");
    std::printf("  --alloc-size <N>              Allocation size in bytes (default: 64)\n");
    std::printf("  --lifecycle                   Run model/session lifecycle bench (vm + arena)\n");
    std::printf("  --lifecycle-iters <N>         Iterations for lifecycle bench (default: 50000)\n");
    std::printf("  --infer                       Run inference benchmark (needs a GGUF)\n");
    std::printf("  --infer-warmup <N>            Warmup tokens before measuring (default: 16)\n");
    std::printf("  --infer-tokens <N>            Tokens to generate for measurement (default: 128)\n");
    std::printf("  --embed                       Run embeddings benchmark (default backend: toy)\n");
    std::printf("  --embed-dim <N>               Embedding dimension override (toy backend)\n");
    std::printf("  --embed-iters <N>             Embeddings iterations (default: 10000)\n");
    std::printf("  --features                    Run feature-surface benches (embeddings/KV/grammar/logprobs)\n");
    std::printf("  --platform                    Run platform primitive microbenchmarks\n");
    std::printf("  --platform-iters <N>          Iterations for platform primitive benches (default: 1000000)\n");
    std::printf("  --spsc-items <N>              Items to transfer (default: 10000000)\n");
    std::printf("  --mpsc-producers <N>          Producer threads (default: 4)\n");
    std::printf("  --mpsc-items <N>              Items per producer (default: 1000000)\n");
    std::printf("  --mpmc-producers <N>          Producer threads (default: 4)\n");
    std::printf("  --mpmc-consumers <N>          Consumer threads (default: 4)\n");
    std::printf("  --mpmc-items <N>              Items per producer (default: 1000000)\n");
    std::printf("  --help                        Show this help\n");
}

} // namespace

int main(int argc, char** argv) {
    Options opt{};

    const uint32_t hw = std::thread::hardware_concurrency();
    if (hw > 0) {
        const uint32_t half = hw / 2;
        opt.mpmc_producers = half > 0 ? half : 1;
        opt.mpmc_consumers = hw - opt.mpmc_producers;
        opt.mpsc_producers = opt.mpmc_producers;
        if (opt.mpmc_consumers == 0) {
            opt.mpmc_consumers = 1;
        }
    }

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (arg == nullptr) {
            continue;
        }

        if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }

        if (std::strcmp(arg, "--only") == 0 && i + 1 < argc) {
            const char* which = argv[++i];
            if (std::strcmp(which, "spsc") == 0) {
                opt.run_spsc = true;
                opt.run_mpsc = false;
                opt.run_mpmc = false;
                opt.run_alloc = false;
                opt.run_lifecycle = false;
                opt.run_infer = false;
                opt.run_embed = false;
                opt.run_features = false;
                opt.run_platform = false;
            } else if (std::strcmp(which, "mpsc") == 0) {
                opt.run_spsc = false;
                opt.run_mpsc = true;
                opt.run_mpmc = false;
                opt.run_alloc = false;
                opt.run_lifecycle = false;
                opt.run_infer = false;
                opt.run_embed = false;
                opt.run_features = false;
                opt.run_platform = false;
            } else if (std::strcmp(which, "mpmc") == 0) {
                opt.run_spsc = false;
                opt.run_mpsc = false;
                opt.run_mpmc = true;
                opt.run_alloc = false;
                opt.run_lifecycle = false;
                opt.run_infer = false;
                opt.run_embed = false;
                opt.run_features = false;
                opt.run_platform = false;
            } else if (std::strcmp(which, "alloc") == 0) {
                opt.run_spsc = false;
                opt.run_mpsc = false;
                opt.run_mpmc = false;
                opt.run_alloc = true;
                opt.run_lifecycle = false;
                opt.run_infer = false;
                opt.run_embed = false;
                opt.run_features = false;
                opt.run_platform = false;
            } else if (std::strcmp(which, "lifecycle") == 0) {
                opt.run_spsc = false;
                opt.run_mpsc = false;
                opt.run_mpmc = false;
                opt.run_alloc = false;
                opt.run_lifecycle = true;
                opt.run_infer = false;
                opt.run_embed = false;
                opt.run_features = false;
                opt.run_platform = false;
            } else if (std::strcmp(which, "infer") == 0) {
                opt.run_spsc = false;
                opt.run_mpsc = false;
                opt.run_mpmc = false;
                opt.run_alloc = false;
                opt.run_lifecycle = false;
                opt.run_infer = true;
                opt.run_embed = false;
                opt.run_features = false;
                opt.run_platform = false;
            } else if (std::strcmp(which, "embed") == 0) {
                opt.run_spsc = false;
                opt.run_mpsc = false;
                opt.run_mpmc = false;
                opt.run_alloc = false;
                opt.run_lifecycle = false;
                opt.run_infer = false;
                opt.run_embed = true;
                opt.run_features = false;
                opt.run_platform = false;
            } else if (std::strcmp(which, "features") == 0) {
                opt.run_spsc = false;
                opt.run_mpsc = false;
                opt.run_mpmc = false;
                opt.run_alloc = false;
                opt.run_lifecycle = false;
                opt.run_infer = false;
                opt.run_embed = false;
                opt.run_features = true;
                opt.run_platform = false;
            } else if (std::strcmp(which, "platform") == 0) {
                opt.run_spsc = false;
                opt.run_mpsc = false;
                opt.run_mpmc = false;
                opt.run_alloc = false;
                opt.run_lifecycle = false;
                opt.run_infer = false;
                opt.run_embed = false;
                opt.run_features = false;
                opt.run_platform = true;
            } else {
                std::fprintf(stderr, "Unknown --only value: %s\n", which);
                return 2;
            }
            continue;
        }

        if (std::strcmp(arg, "--spsc-items") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], &opt.spsc_items)) {
                std::fprintf(stderr, "Invalid --spsc-items value\n");
                return 2;
            }
            continue;
        }

        if (std::strcmp(arg, "--mpsc-producers") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &opt.mpsc_producers)) {
                std::fprintf(stderr, "Invalid --mpsc-producers value\n");
                return 2;
            }
            continue;
        }

        if (std::strcmp(arg, "--mpsc-items") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], &opt.mpsc_items_per_producer)) {
                std::fprintf(stderr, "Invalid --mpsc-items value\n");
                return 2;
            }
            continue;
        }

        if (std::strcmp(arg, "--mpmc-producers") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &opt.mpmc_producers)) {
                std::fprintf(stderr, "Invalid --mpmc-producers value\n");
                return 2;
            }
            continue;
        }

        if (std::strcmp(arg, "--mpmc-consumers") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &opt.mpmc_consumers)) {
                std::fprintf(stderr, "Invalid --mpmc-consumers value\n");
                return 2;
            }
            continue;
        }

        if (std::strcmp(arg, "--mpmc-items") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], &opt.mpmc_items_per_producer)) {
                std::fprintf(stderr, "Invalid --mpmc-items value\n");
                return 2;
            }
            continue;
        }

        if (std::strcmp(arg, "--infer") == 0) {
            opt.run_infer = true;
            continue;
        }

        if (std::strcmp(arg, "--alloc") == 0) {
            opt.run_alloc = true;
            continue;
        }

        if (std::strcmp(arg, "--lifecycle") == 0) {
            opt.run_lifecycle = true;
            continue;
        }

        if (std::strcmp(arg, "--features") == 0) {
            opt.run_features = true;
            continue;
        }

        if (std::strcmp(arg, "--platform") == 0) {
            opt.run_platform = true;
            continue;
        }

        if (std::strcmp(arg, "--platform-iters") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], &opt.platform_iters)) {
                std::fprintf(stderr, "Invalid --platform-iters value\n");
                return 2;
            }
            continue;
        }

        if (std::strcmp(arg, "--embed") == 0) {
            opt.run_embed = true;
            continue;
        }

        if (std::strcmp(arg, "--alloc-iters") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], &opt.alloc_iters)) {
                std::fprintf(stderr, "Invalid --alloc-iters value\n");
                return 2;
            }
            continue;
        }

        if (std::strcmp(arg, "--alloc-size") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &opt.alloc_size)) {
                std::fprintf(stderr, "Invalid --alloc-size value\n");
                return 2;
            }
            continue;
        }

        if (std::strcmp(arg, "--lifecycle-iters") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], &opt.lifecycle_iters)) {
                std::fprintf(stderr, "Invalid --lifecycle-iters value\n");
                return 2;
            }
            continue;
        }

        if (std::strcmp(arg, "--embed-dim") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &opt.embed_dim)) {
                std::fprintf(stderr, "Invalid --embed-dim value\n");
                return 2;
            }
            continue;
        }

        if (std::strcmp(arg, "--embed-iters") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], &opt.embed_iters)) {
                std::fprintf(stderr, "Invalid --embed-iters value\n");
                return 2;
            }
            continue;
        }

        if (std::strcmp(arg, "--infer-warmup") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &opt.infer_warmup_tokens)) {
                std::fprintf(stderr, "Invalid --infer-warmup value\n");
                return 2;
            }
            continue;
        }

        if (std::strcmp(arg, "--infer-tokens") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &opt.infer_tokens)) {
                std::fprintf(stderr, "Invalid --infer-tokens value\n");
                return 2;
            }
            continue;
        }

        std::fprintf(stderr, "Unknown option: %s\n", arg);
        print_usage(argv[0]);
        return 2;
    }

    const astral::bench::ClockInfo clk = astral::bench::clock_info();
    std::printf("Astral benchmarks (ticks: %s)\n", clk.name);
    std::printf("%-28s  %8s  %10s  %12s\n", "Benchmark", "Mops/s", "ns/op", "ticks/op");

    if (opt.run_spsc) {
        const auto r = astral::bench::bench_spsc_ring(opt.spsc_items);
        astral::bench::print_result(r, clk.name);
    }

    if (opt.run_mpsc) {
        const auto r = astral::bench::bench_mpsc_ring(opt.mpsc_producers, opt.mpsc_items_per_producer);
        astral::bench::print_result(r, clk.name);
    }

    if (opt.run_mpmc) {
        const auto r = astral::bench::bench_mpmc_queue(opt.mpmc_producers, opt.mpmc_consumers,
                                                       opt.mpmc_items_per_producer);
        astral::bench::print_result(r, clk.name);
    }

    if (opt.run_alloc) {
        astral::bench::print_result(astral::bench::bench_runtime_alloc_free(opt.alloc_iters, opt.alloc_size, false), clk.name);
        astral::bench::print_result(astral::bench::bench_runtime_alloc_free(opt.alloc_iters, opt.alloc_size, true), clk.name);
    }

    if (opt.run_lifecycle) {
        astral::bench::print_result(astral::bench::bench_model_load_release(opt.lifecycle_iters, false), clk.name);
        astral::bench::print_result(astral::bench::bench_model_load_release(opt.lifecycle_iters, true), clk.name);
        astral::bench::print_result(astral::bench::bench_session_create_destroy(opt.lifecycle_iters, false), clk.name);
        astral::bench::print_result(astral::bench::bench_session_create_destroy(opt.lifecycle_iters, true), clk.name);
    }

    if (opt.run_infer) {
        astral::bench::bench_inference_print(opt.infer_warmup_tokens, opt.infer_tokens);
    }

    if (opt.run_embed) {
        astral::bench::bench_embeddings_print(opt.embed_dim, opt.embed_iters);
    }

    if (opt.run_features) {
        astral::bench::bench_feature_surfaces_print();
    }

    if (opt.run_platform) {
        astral::bench::bench_platform_print(opt.platform_iters);
    }

    return 0;
}
