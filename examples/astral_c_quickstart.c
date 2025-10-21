#include "astral_rt.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static AstralSpanU8 span_from_cstr(const char* value) {
    AstralSpanU8 span = {0};
    if (value == NULL) {
        return span;
    }
    span.data = (const uint8_t*)value;
    span.len = (uint32_t)strlen(value);
    return span;
}

static uint32_t parse_u32(const char* value, uint32_t fallback) {
    char* end = NULL;
    unsigned long parsed = 0;
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    parsed = strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed > UINT32_MAX) {
        return fallback;
    }
    return (uint32_t)parsed;
}

static void print_err(const char* label, AstralErr err) {
    fprintf(stderr, "%s failed: %s (%s)\n", label, astral_error_string(err), astral_last_error());
}

static void usage(const char* argv0) {
    fprintf(stderr,
            "Usage: %s [--backend mock|cpu|cuda] [--model path.gguf] [--prompt text] [--max-tokens n]\n",
            argv0);
}

int main(int argc, char** argv) {
    const char* backend = "mock";
    const char* model_path = "infinite";
    const char* prompt = "Once upon a time";
    uint32_t max_tokens = 32;
    AstralHandle model = 0;
    AstralHandle session = 0;
    int rc = 1;
    AstralInit init = {0};
    AstralModelDesc model_desc = {0};
    AstralSessionDesc session_desc = {0};
    AstralStats stats = {0};
    AstralErr err = ASTRAL_OK;
    uint8_t buffer[512];

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = parse_u32(argv[++i], max_tokens);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    init.reserve_bytes = 512ULL * 1024ULL * 1024ULL;
    init.thread_count = 0;
    init.numa_node = 0xFFFFFFFFu;
    init.enable_hugepages = 0;

    err = astral_init(&init);
    if (err != ASTRAL_OK) {
        print_err("astral_init", err);
        goto out;
    }

    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    model_desc.model_path = span_from_cstr(model_path);
    model_desc.backend_name = span_from_cstr(backend);
    model_desc.gpu_layers = 0;
    model_desc.n_ctx = 512;
    model_desc.n_batch = 128;
    model_desc.n_threads = 0;
    model_desc.embeddings_only = 0;

    err = astral_model_load(&model_desc, &model);
    if (err != ASTRAL_OK) {
        print_err("astral_model_load", err);
        goto out;
    }

    session_desc.model = model;
    session_desc.max_tokens = max_tokens;
    session_desc.temperature = 0.7f;
    session_desc.top_k = 40;
    session_desc.top_p = 0.95f;
    session_desc.stream_enabled = 1;
    session_desc.seed = 1;

    err = astral_session_create(&session_desc, &session);
    if (err != ASTRAL_OK) {
        print_err("astral_session_create", err);
        goto out;
    }

    err = astral_session_feed(session, span_from_cstr(prompt), 1);
    if (err != ASTRAL_OK) {
        print_err("astral_session_feed", err);
        goto out;
    }

    err = astral_session_decode(session);
    if (err != ASTRAL_OK) {
        print_err("astral_session_decode", err);
        goto out;
    }

    for (;;) {
        AstralMutSpanU8 out_span = {0};
        int32_t bytes_read = 0;
        out_span.data = buffer;
        out_span.len = (uint32_t)sizeof(buffer);

        bytes_read = astral_stream_read(session, out_span, 100);
        if (bytes_read == ASTRAL_E_TIMEOUT) {
            continue;
        }
        if (bytes_read < 0) {
            fprintf(stderr, "astral_stream_read failed: %d (%s)\n", bytes_read, astral_last_error());
            goto out;
        }
        if (bytes_read == 0) {
            break;
        }
        fwrite(buffer, 1, (size_t)bytes_read, stdout);
    }

    err = astral_session_wait(session, 60000);
    if (err != ASTRAL_OK) {
        print_err("astral_session_wait", err);
        goto out;
    }

    err = astral_session_stats(session, &stats);
    if (err == ASTRAL_OK) {
        fprintf(stderr, "\ntok/s=%.2f ttft_ms=%.2f\n", stats.tok_per_s, stats.t_first_token_ms);
    }

    session_desc.seed = 2;
    err = astral_session_reset(session, &session_desc);
    if (err != ASTRAL_OK) {
        print_err("astral_session_reset", err);
        goto out;
    }

    rc = 0;

out:
    if (session != 0) {
        astral_session_destroy(session);
    }
    if (model != 0) {
        astral_model_release(model);
    }
    astral_shutdown();
    return rc;
}
