#include "bench_clock.hpp"
#include "bench_common.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace astral::bench {
BenchResult bench_spsc_ring(uint64_t items);
BenchResult bench_mpmc_queue(uint32_t producers, uint32_t consumers, uint64_t items_per_producer);
void bench_inference_print(uint32_t warmup_tokens, uint32_t measure_tokens);
void bench_embeddings_print(uint32_t dim_override, uint64_t iters);
} // namespace astral::bench

namespace {

struct Options {
    uint64_t spsc_items = 10'000'000ull;
    uint32_t mpmc_producers = 4;
    uint32_t mpmc_consumers = 4;
    uint64_t mpmc_items_per_producer = 1'000'000ull;
    bool run_spsc = true;
    bool run_mpmc = true;
    bool run_infer = false;
    uint32_t infer_warmup_tokens = 16;
    uint32_t infer_tokens = 128;
    bool run_embed = false;
    uint32_t embed_dim = 0;
    uint64_t embed_iters = 10000;
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
    std::printf("  --only <spsc|mpmc|infer|embed> Run only one benchmark\n");
    std::printf("  --infer                       Run inference benchmark (needs a GGUF)\n");
    std::printf("  --infer-warmup <N>            Warmup tokens before measuring (default: 16)\n");
    std::printf("  --infer-tokens <N>            Tokens to generate for measurement (default: 128)\n");
    std::printf("  --embed                       Run embeddings benchmark (default backend: toy)\n");
    std::printf("  --embed-dim <N>               Embedding dimension override (toy backend)\n");
    std::printf("  --embed-iters <N>             Embeddings iterations (default: 10000)\n");
    std::printf("  --spsc-items <N>              Items to transfer (default: 10000000)\n");
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
                opt.run_mpmc = false;
                opt.run_infer = false;
                opt.run_embed = false;
            } else if (std::strcmp(which, "mpmc") == 0) {
                opt.run_spsc = false;
                opt.run_mpmc = true;
                opt.run_infer = false;
                opt.run_embed = false;
            } else if (std::strcmp(which, "infer") == 0) {
                opt.run_spsc = false;
                opt.run_mpmc = false;
                opt.run_infer = true;
                opt.run_embed = false;
            } else if (std::strcmp(which, "embed") == 0) {
                opt.run_spsc = false;
                opt.run_mpmc = false;
                opt.run_infer = false;
                opt.run_embed = true;
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

        if (std::strcmp(arg, "--embed") == 0) {
            opt.run_embed = true;
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

    if (opt.run_mpmc) {
        const auto r = astral::bench::bench_mpmc_queue(opt.mpmc_producers, opt.mpmc_consumers,
                                                       opt.mpmc_items_per_producer);
        astral::bench::print_result(r, clk.name);
    }

    if (opt.run_infer) {
        astral::bench::bench_inference_print(opt.infer_warmup_tokens, opt.infer_tokens);
    }

    if (opt.run_embed) {
        astral::bench::bench_embeddings_print(opt.embed_dim, opt.embed_iters);
    }

    return 0;
}
