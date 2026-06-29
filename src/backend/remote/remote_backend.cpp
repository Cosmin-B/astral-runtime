#include "../backend.hpp"
#include "../../core/runtime_alloc.hpp"
#include "../../utils/trace.hpp"

#include <algorithm>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <thread>

#include <cpp-httplib/httplib.h>

namespace astral::backend {
namespace {

constexpr uint32_t kRemoteVocabSize = 258;
constexpr int32_t kRemoteTokenBos = 256;
constexpr int32_t kRemoteTokenEos = 257;
constexpr uint32_t kRemoteDefaultCtx = 4096;
constexpr uint32_t kRemoteDefaultEmbeddingDim = 8;
constexpr uint32_t kRemoteMaxPromptBytes = 8192;
constexpr uint32_t kRemoteMaxOutputBytes = 8192;
constexpr uint32_t kRemoteMaxUrlBytes = 512;
constexpr uint32_t kRemoteMaxApiKeyBytes = 256;
constexpr uint32_t kRemoteMaxStreamLineBytes = 4096;
constexpr uint32_t kRemoteTimeoutSeconds = 5;
constexpr uint32_t kRemoteMaxAttempts = 2;
constexpr float kRemoteSelectedLogit = 1000.0f;
constexpr float kRemoteSuppressedLogit = -1000.0f;
constexpr char kRemoteSseDataPrefix[] = "data:";
constexpr char kRemoteSseDone[] = "[DONE]";
constexpr char kRemoteJsonContentKey[] = "\"content\"";
constexpr char kRemoteJsonTextKey[] = "\"text\"";

struct RemoteModel {
    char base_url[kRemoteMaxUrlBytes];
    char api_key[kRemoteMaxApiKeyBytes];
    uint32_t ctx_size;
    uint32_t embedding_dim;
};

struct RemoteSession {
    RemoteModel* model = nullptr;
    std::mutex stream_mutex;
    std::condition_variable stream_cv;
    std::thread stream_thread;
    uint8_t prompt[kRemoteMaxPromptBytes];
    uint8_t output[kRemoteMaxOutputBytes];
    char stream_line[kRemoteMaxStreamLineBytes];
    float logits[kRemoteVocabSize];
    uint32_t prompt_len = 0;
    uint32_t output_len = 0;
    uint32_t output_pos = 0;
    uint32_t stream_line_len = 0;
    uint32_t last_batch_outputs = 0;
    AstralErr stream_err = ASTRAL_OK;
    bool stream_passthrough = false;
    bool stream_started = false;
    bool stream_done = false;
    bool stream_sse_mode = false;
};

struct RemoteEmbedder {
    RemoteModel* model;
};

static bool span_copy_nt(AstralSpanU8 span, char* dst, uint32_t cap) {
    if (dst == nullptr || cap == 0 || span.data == nullptr || span.len == 0 || span.len >= cap) {
        return false;
    }
    std::memcpy(dst, span.data, span.len);
    dst[span.len] = '\0';
    return true;
}

static bool is_http_url(const char* url) {
    return std::strncmp(url, "http://", 7) == 0 || std::strncmp(url, "https://", 8) == 0;
}

static bool is_https_url(const char* url) {
    return std::strncmp(url, "https://", 8) == 0;
}

static bool remote_tls_available() {
#if defined(CPPHTTPLIB_OPENSSL_SUPPORT)
    return true;
#else
    return false;
#endif
}

static httplib::Headers remote_headers(const RemoteModel* model) {
    httplib::Headers headers;
    if (model != nullptr && model->api_key[0] != '\0') {
        headers.emplace("Authorization", std::string("Bearer ") + model->api_key);
    }
    return headers;
}

static AstralErr http_status_to_err(int status) {
    if (status == 200) {
        return ASTRAL_OK;
    }
    if (status == 404) {
        return ASTRAL_E_NOT_FOUND;
    }
    if (status == 408 || status == 504) {
        return ASTRAL_E_TIMEOUT;
    }
    if (status == 501 || status == 405) {
        return ASTRAL_E_UNSUPPORTED;
    }
    return ASTRAL_E_BACKEND;
}

static bool http_status_retryable(int status) {
    return status == 408 || status == 429 || status == 500 || status == 502 || status == 503 || status == 504;
}

static AstralErr post_remote(const RemoteModel* model, const char* path, const char* body, uint32_t body_len, std::string* out_body) {
    ASTRAL_ZONE_N("astral.remote.transport_wait");
    if (model == nullptr || path == nullptr || out_body == nullptr) {
        return ASTRAL_E_INVALID;
    }

    httplib::Client client(model->base_url);
    client.set_connection_timeout(kRemoteTimeoutSeconds, 0);
    client.set_read_timeout(kRemoteTimeoutSeconds, 0);
    client.set_write_timeout(kRemoteTimeoutSeconds, 0);

    const std::string payload(body != nullptr ? body : "", body_len);
    AstralErr last_err = ASTRAL_E_TIMEOUT;
    for (uint32_t attempt = 0; attempt < kRemoteMaxAttempts; ++attempt) {
        auto result = client.Post(path, remote_headers(model), payload, "text/plain");
        if (!result) {
            last_err = ASTRAL_E_TIMEOUT;
            continue;
        }
        const AstralErr status_err = http_status_to_err(result->status);
        if (status_err == ASTRAL_OK) {
            *out_body = result->body;
            return ASTRAL_OK;
        }
        last_err = status_err;
        if (!http_status_retryable(result->status)) {
            return last_err;
        }
    }
    return last_err;
}

static AstralErr get_remote(const RemoteModel* model, const char* path) {
    ASTRAL_ZONE_N("astral.remote.transport_wait");
    if (model == nullptr || path == nullptr) {
        return ASTRAL_E_INVALID;
    }

    httplib::Client client(model->base_url);
    client.set_connection_timeout(kRemoteTimeoutSeconds, 0);
    client.set_read_timeout(kRemoteTimeoutSeconds, 0);
    client.set_write_timeout(kRemoteTimeoutSeconds, 0);

    AstralErr last_err = ASTRAL_E_TIMEOUT;
    for (uint32_t attempt = 0; attempt < kRemoteMaxAttempts; ++attempt) {
        auto result = client.Get(path, remote_headers(model));
        if (!result) {
            last_err = ASTRAL_E_TIMEOUT;
            continue;
        }
        last_err = http_status_to_err(result->status);
        if (last_err == ASTRAL_OK || !http_status_retryable(result->status)) {
            return last_err;
        }
    }
    return last_err;
}

static uint32_t parse_token_list(const char* text, uint32_t len, int32_t* out_tokens, uint32_t max_tokens) {
    uint32_t count = 0;
    const char* p = text;
    const char* end = text + len;
    while (p < end) {
        while (p < end && !std::isdigit(static_cast<unsigned char>(*p)) && *p != '-') {
            ++p;
        }
        if (p >= end) {
            break;
        }
        char* next = nullptr;
        const long v = std::strtol(p, &next, 10);
        if (next == p) {
            break;
        }
        if (out_tokens != nullptr && count < max_tokens) {
            out_tokens[count] = static_cast<int32_t>(v);
        }
        ++count;
        p = next;
    }
    return count;
}

static uint32_t parse_float_list(const char* text, uint32_t len, float* out_vec, uint32_t max_values) {
    uint32_t count = 0;
    const char* p = text;
    const char* end = text + len;
    while (p < end) {
        while (p < end && !std::isdigit(static_cast<unsigned char>(*p)) && *p != '-' && *p != '+') {
            ++p;
        }
        if (p >= end) {
            break;
        }
        char* next = nullptr;
        const float v = std::strtof(p, &next);
        if (next == p) {
            break;
        }
        if (out_vec != nullptr && count < max_values) {
            out_vec[count] = v;
        }
        ++count;
        p = next;
    }
    return count;
}

static uint32_t clamp_size_to_u32(std::size_t value) {
    return value > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max()
                                                        : static_cast<uint32_t>(value);
}

static AstralErr remote_stream_set_error(RemoteSession* session, AstralErr err) {
  if (session == nullptr) {
    return ASTRAL_E_INVALID;
  }
  std::unique_lock<std::mutex> lock(session->stream_mutex);
  session->stream_err = err;
  session->stream_done = true;
  lock.unlock();
  session->stream_cv.notify_all();
  return err;
}

static AstralErr remote_stream_append_raw(RemoteSession* session, const char* data, uint32_t len) {
  if (session == nullptr || data == nullptr || len == 0) {
    return ASTRAL_OK;
  }
  std::unique_lock<std::mutex> lock(session->stream_mutex);
  const uint32_t room = kRemoteMaxOutputBytes - session->output_len;
  if (len > room) {
    lock.unlock();
    return remote_stream_set_error(session, ASTRAL_E_NOMEM);
  }
  std::memcpy(session->output + session->output_len, data, len);
  session->output_len += len;
  lock.unlock();
  session->stream_cv.notify_all();
  return ASTRAL_OK;
}

static bool remote_stream_line_matches_prefix(const char* line, uint32_t len) {
  const uint32_t prefix_len = static_cast<uint32_t>(sizeof(kRemoteSseDataPrefix) - 1);
  return len <= prefix_len && std::memcmp(line, kRemoteSseDataPrefix, len) == 0;
}

static const char* remote_find_bytes(const char* haystack, uint32_t haystack_len,
                                     const char* needle, uint32_t needle_len) {
  if (needle_len == 0 || haystack_len < needle_len) {
    return nullptr;
  }
  const uint32_t limit = haystack_len - needle_len;
  for (uint32_t i = 0; i <= limit; ++i) {
    if (std::memcmp(haystack + i, needle, needle_len) == 0) {
      return haystack + i;
    }
  }
  return nullptr;
}

static const char* remote_json_string_value(const char* payload, uint32_t payload_len,
                                            const char* key, uint32_t key_len, uint32_t* out_len) {
  const char* key_pos = remote_find_bytes(payload, payload_len, key, key_len);
  if (key_pos == nullptr) {
    return nullptr;
  }

  const char* cursor = key_pos + key_len;
  const char* end = payload + payload_len;
  while (cursor < end && std::isspace(static_cast<unsigned char>(*cursor))) {
    ++cursor;
  }
  if (cursor == end || *cursor != ':') {
    return nullptr;
  }
  ++cursor;
  while (cursor < end && std::isspace(static_cast<unsigned char>(*cursor))) {
    ++cursor;
  }
  if (cursor == end || *cursor != '"') {
    return nullptr;
  }
  ++cursor;

  const char* value = cursor;
  uint32_t len = 0;
  bool escaped = false;
  while (cursor < end) {
    const char c = *cursor++;
    if (escaped) {
      escaped = false;
      ++len;
      continue;
    }
    if (c == '\\') {
      escaped = true;
      ++len;
      continue;
    }
    if (c == '"') {
      *out_len = len;
      return value;
    }
    ++len;
  }
  return nullptr;
}

static AstralErr remote_stream_append_json_string(RemoteSession* session, const char* value,
                                                  uint32_t len) {
  char decoded[kRemoteMaxStreamLineBytes];
  uint32_t decoded_len = 0;
  for (uint32_t i = 0; i < len && decoded_len < kRemoteMaxStreamLineBytes; ++i) {
    char c = value[i];
    if (c == '\\' && i + 1u < len) {
      c = value[++i];
      switch (c) {
      case 'n':
        c = '\n';
        break;
      case 'r':
        c = '\r';
        break;
      case 't':
        c = '\t';
        break;
      case 'b':
        c = '\b';
        break;
      case 'f':
        c = '\f';
        break;
      case 'u':
        if (i + 4u < len) {
          i += 4u;
        }
        c = '?';
        break;
      default:
        break;
      }
    }
    decoded[decoded_len++] = c;
  }
  return decoded_len != 0 ? remote_stream_append_raw(session, decoded, decoded_len) : ASTRAL_OK;
}

static AstralErr remote_stream_append_sse_payload(RemoteSession* session, const char* payload,
                                                  uint32_t payload_len) {
  const uint32_t content_key_len = static_cast<uint32_t>(sizeof(kRemoteJsonContentKey) - 1);
  const uint32_t text_key_len = static_cast<uint32_t>(sizeof(kRemoteJsonTextKey) - 1);
  uint32_t value_len = 0;
  const char* value = remote_json_string_value(payload, payload_len, kRemoteJsonContentKey,
                                               content_key_len, &value_len);
  if (value == nullptr) {
    value = remote_json_string_value(payload, payload_len, kRemoteJsonTextKey, text_key_len,
                                     &value_len);
  }
  return value != nullptr ? remote_stream_append_json_string(session, value, value_len)
                          : remote_stream_append_raw(session, payload, payload_len);
}

static AstralErr remote_stream_process_sse_line(RemoteSession* session) {
  uint32_t len = session->stream_line_len;
  if (len != 0 && session->stream_line[len - 1] == '\n') {
    --len;
  }
  if (len != 0 && session->stream_line[len - 1] == '\r') {
    --len;
  }

  const uint32_t prefix_len = static_cast<uint32_t>(sizeof(kRemoteSseDataPrefix) - 1);
  if (len >= prefix_len &&
      std::memcmp(session->stream_line, kRemoteSseDataPrefix, prefix_len) == 0) {
    const char* payload = session->stream_line + prefix_len;
    uint32_t payload_len = len - prefix_len;
    while (payload_len != 0 && *payload == ' ') {
      ++payload;
      --payload_len;
    }
    const uint32_t done_len = static_cast<uint32_t>(sizeof(kRemoteSseDone) - 1);
    if (payload_len != 0 &&
        (payload_len != done_len || std::memcmp(payload, kRemoteSseDone, done_len) != 0)) {
      const AstralErr err = remote_stream_append_sse_payload(session, payload, payload_len);
      if (err != ASTRAL_OK) {
        session->stream_line_len = 0;
        return err;
      }
    }
  }
  session->stream_line_len = 0;
  return ASTRAL_OK;
}

static AstralErr remote_stream_append_sse_aware(RemoteSession* session, const char* data,
                                                uint32_t len) {
  if (session == nullptr || data == nullptr || len == 0) {
    return ASTRAL_OK;
  }

  uint32_t offset = 0;
  while (offset < len) {
    if (session->stream_passthrough) {
      return remote_stream_append_raw(session, data + offset, len - offset);
    }

    if (session->stream_line_len == kRemoteMaxStreamLineBytes) {
      return remote_stream_set_error(session, ASTRAL_E_NOMEM);
    }

    const char c = data[offset++];
    session->stream_line[session->stream_line_len++] = c;

    if (!session->stream_sse_mode) {
      if (!remote_stream_line_matches_prefix(session->stream_line, session->stream_line_len)) {
        session->stream_passthrough = true;
        const AstralErr err =
            remote_stream_append_raw(session, session->stream_line, session->stream_line_len);
        session->stream_line_len = 0;
        if (err != ASTRAL_OK) {
          return err;
        }
      } else if (session->stream_line_len ==
                 static_cast<uint32_t>(sizeof(kRemoteSseDataPrefix) - 1)) {
        session->stream_sse_mode = true;
      }
      continue;
    }

    if (c == '\n') {
      const AstralErr err = remote_stream_process_sse_line(session);
      if (err != ASTRAL_OK) {
        return err;
      }
    }
  }
  return ASTRAL_OK;
}

static AstralErr remote_stream_flush_parser(RemoteSession* session) {
  if (session == nullptr || session->stream_line_len == 0) {
    return ASTRAL_OK;
  }
  if (session->stream_sse_mode) {
    return remote_stream_process_sse_line(session);
  }
  const AstralErr err =
      remote_stream_append_raw(session, session->stream_line, session->stream_line_len);
  session->stream_line_len = 0;
  return err;
}

static void remote_stream_finish(RemoteSession* session, AstralErr err) {
  if (session == nullptr) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(session->stream_mutex);
    if (session->stream_err == ASTRAL_OK || err != ASTRAL_OK) {
      session->stream_err = err;
    }
    session->stream_done = true;
  }
  session->stream_cv.notify_all();
}

static AstralErr remote_stream_error(RemoteSession* session) {
  if (session == nullptr) {
    return ASTRAL_E_INVALID;
  }
  std::lock_guard<std::mutex> lock(session->stream_mutex);
  return session->stream_err;
}

struct RemoteStreamReceiver {
  RemoteSession* session;

  bool operator()(const char* data, size_t len) const {
    return len == 0 ||
           remote_stream_append_sse_aware(session, data, clamp_size_to_u32(len)) == ASTRAL_OK;
  }
};

static AstralErr post_remote_stream(RemoteSession* session, const char* body, uint32_t body_len) {
  ASTRAL_ZONE_N("astral.remote.stream_completion");
  if (session == nullptr || session->model == nullptr) {
    return ASTRAL_E_INVALID;
  }

  httplib::Client client(session->model->base_url);
  client.set_connection_timeout(kRemoteTimeoutSeconds, 0);
  client.set_read_timeout(kRemoteTimeoutSeconds, 0);
  client.set_write_timeout(kRemoteTimeoutSeconds, 0);

  const std::string payload(body != nullptr ? body : "", body_len);
  AstralErr last_err = ASTRAL_E_TIMEOUT;
  RemoteStreamReceiver receiver{session};
  for (uint32_t attempt = 0; attempt < kRemoteMaxAttempts; ++attempt) {
    auto result = client.Post("/completion/stream", remote_headers(session->model), payload,
                              "text/plain", receiver);
    const AstralErr stream_err = remote_stream_error(session);
    if (stream_err != ASTRAL_OK) {
      return stream_err;
    }
    if (!result) {
      last_err = ASTRAL_E_TIMEOUT;
      continue;
    }
    const AstralErr status_err = http_status_to_err(result->status);
    if (status_err == ASTRAL_OK) {
      return remote_stream_flush_parser(session);
    }
    last_err = status_err;
    if (!http_status_retryable(result->status)) {
      return last_err;
    }
  }
  return last_err;
}

static void remote_completion_worker(RemoteSession* session) {
  AstralErr err = post_remote_stream(session, reinterpret_cast<const char*>(session->prompt),
                                     session->prompt_len);
  if (err == ASTRAL_E_NOT_FOUND || err == ASTRAL_E_UNSUPPORTED) {
    std::string body;
    err = post_remote(session->model, "/completion", reinterpret_cast<const char*>(session->prompt),
                      session->prompt_len, &body);
    if (err == ASTRAL_OK) {
      err = remote_stream_append_raw(session, body.data(), clamp_size_to_u32(body.size()));
    }
  }
  remote_stream_finish(session, err);
}

static AstralErr remote_session_start_stream(RemoteSession* session) {
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }
    {
        std::lock_guard<std::mutex> lock(session->stream_mutex);
        if (session->stream_started) {
            return ASTRAL_OK;
        }
        session->output_len = 0;
        session->output_pos = 0;
        session->stream_line_len = 0;
        session->stream_err = ASTRAL_OK;
        session->stream_passthrough = false;
        session->stream_done = false;
        session->stream_started = true;
        session->stream_sse_mode = false;
    }
    try {
        session->stream_thread = std::thread(remote_completion_worker, session);
    } catch (...) {
        remote_stream_finish(session, ASTRAL_E_NOMEM);
        return ASTRAL_E_NOMEM;
    }
    return ASTRAL_OK;
}

static void remote_session_join_stream(RemoteSession* session) {
    if (session != nullptr && session->stream_thread.joinable()) {
        session->stream_thread.join();
    }
}

void* remote_model_load(const AstralModelDesc* desc, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }
    if (desc == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }

    RemoteModel* model = core::runtime_new<RemoteModel>();
    if (model == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }
    std::memset(model, 0, sizeof(*model));

    if (!span_copy_nt(desc->model_path, model->base_url, sizeof(model->base_url)) || !is_http_url(model->base_url)) {
        core::runtime_delete(model);
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }
    if (is_https_url(model->base_url) && !remote_tls_available()) {
        core::runtime_delete(model);
        *out_err = ASTRAL_E_UNSUPPORTED;
        return nullptr;
    }

    if (desc->model_bytes.data != nullptr && desc->model_bytes.len != 0) {
        if (!span_copy_nt(desc->model_bytes, model->api_key, sizeof(model->api_key))) {
            core::runtime_delete(model);
            *out_err = ASTRAL_E_INVALID;
            return nullptr;
        }
    }

    model->ctx_size = desc->n_ctx != 0 ? desc->n_ctx : kRemoteDefaultCtx;
    model->embedding_dim = kRemoteDefaultEmbeddingDim;

    const AstralErr health = get_remote(model, "/health");
    if (health != ASTRAL_OK) {
        core::runtime_delete(model);
        *out_err = health;
        return nullptr;
    }

    *out_err = ASTRAL_OK;
    return model;
}

void remote_model_unload(void* model_ctx) {
    core::runtime_delete(static_cast<RemoteModel*>(model_ctx));
}

AstralErr remote_tokenize(void* model_ctx, AstralSpanU8 text, int32_t* out_tokens, uint32_t max_tokens,
                          uint8_t add_special, uint8_t parse_special, uint32_t* out_count) {
    (void)parse_special;
    RemoteModel* model = static_cast<RemoteModel*>(model_ctx);
    if (model == nullptr || out_count == nullptr || (text.len != 0 && text.data == nullptr)) {
        return ASTRAL_E_INVALID;
    }

    std::string body;
    AstralErr err = post_remote(model, "/tokenize", reinterpret_cast<const char*>(text.data), text.len, &body);
    if (err == ASTRAL_E_NOT_FOUND || err == ASTRAL_E_UNSUPPORTED) {
        const uint32_t needed = text.len + (add_special != 0 ? 1u : 0u);
        *out_count = needed;
        if (out_tokens == nullptr && max_tokens == 0) {
            return ASTRAL_OK;
        }
        if (needed > max_tokens) {
            return ASTRAL_E_NOMEM;
        }
        uint32_t n = 0;
        if (add_special != 0) {
            out_tokens[n++] = kRemoteTokenBos;
        }
        for (uint32_t i = 0; i < text.len; ++i) {
            out_tokens[n++] = static_cast<int32_t>(text.data[i]);
        }
        return ASTRAL_OK;
    }
    if (err != ASTRAL_OK) {
        return err;
    }

    const uint32_t remote_count = parse_token_list(body.data(), static_cast<uint32_t>(body.size()), out_tokens, max_tokens);
    const uint32_t needed = remote_count + (add_special != 0 ? 1u : 0u);
    *out_count = needed;
    if (out_tokens == nullptr && max_tokens == 0) {
        return ASTRAL_OK;
    }
    if (needed > max_tokens) {
        return ASTRAL_E_NOMEM;
    }
    if (add_special != 0) {
        for (uint32_t i = remote_count; i > 0; --i) {
            out_tokens[i] = out_tokens[i - 1u];
        }
        out_tokens[0] = kRemoteTokenBos;
    }
    return ASTRAL_OK;
}

AstralErr remote_detokenize(void* model_ctx, const int32_t* tokens, uint32_t count, AstralMutSpanU8 out_text, uint32_t* out_len) {
    (void)model_ctx;
    if (out_len == nullptr || (count != 0 && tokens == nullptr)) {
        return ASTRAL_E_INVALID;
    }
    uint32_t needed = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const int32_t token = tokens[i];
        needed += (token >= 0 && token < 256) ? 1u : 0u;
    }
    *out_len = needed;
    if (out_text.data == nullptr && out_text.len == 0) {
        return ASTRAL_OK;
    }
    if (out_text.data == nullptr || out_text.len < needed) {
        return ASTRAL_E_NOMEM;
    }
    uint32_t n = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const int32_t token = tokens[i];
        if (token >= 0 && token < 256) {
            out_text.data[n++] = static_cast<uint8_t>(token);
        }
    }
    return ASTRAL_OK;
}

AstralErr remote_model_info(void* model_ctx, uint32_t* out_vocab_size, uint32_t* out_ctx_size) {
    RemoteModel* model = static_cast<RemoteModel*>(model_ctx);
    if (model == nullptr || out_vocab_size == nullptr || out_ctx_size == nullptr) {
        return ASTRAL_E_INVALID;
    }
    *out_vocab_size = kRemoteVocabSize;
    *out_ctx_size = model->ctx_size;
    return ASTRAL_OK;
}

AstralErr remote_model_special_tokens(void* model_ctx, int32_t* out_bos, int32_t* out_eos) {
    (void)model_ctx;
    if (out_bos == nullptr || out_eos == nullptr) {
        return ASTRAL_E_INVALID;
    }
    *out_bos = kRemoteTokenBos;
    *out_eos = kRemoteTokenEos;
    return ASTRAL_OK;
}

AstralErr remote_model_embedding_dim(void* model_ctx, uint32_t* out_dim) {
    RemoteModel* model = static_cast<RemoteModel*>(model_ctx);
    if (model == nullptr || out_dim == nullptr) {
        return ASTRAL_E_INVALID;
    }
    *out_dim = model->embedding_dim;
    return ASTRAL_OK;
}

void* remote_session_create(void* model_ctx, const AstralSessionDesc* desc, AstralErr* out_err) {
    (void)desc;
    if (out_err == nullptr) {
        return nullptr;
    }
    RemoteModel* model = static_cast<RemoteModel*>(model_ctx);
    if (model == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }
    RemoteSession* session = core::runtime_new<RemoteSession>();
    if (session == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }
    std::memset(session->prompt, 0, sizeof(session->prompt));
    std::memset(session->output, 0, sizeof(session->output));
    std::memset(session->logits, 0, sizeof(session->logits));
    session->model = model;
    *out_err = ASTRAL_OK;
    return session;
}

void remote_session_destroy(void* session_ctx) {
    RemoteSession* session = static_cast<RemoteSession*>(session_ctx);
    remote_session_join_stream(session);
    core::runtime_delete(session);
}

AstralErr remote_session_reset(void* session_ctx) {
    RemoteSession* session = static_cast<RemoteSession*>(session_ctx);
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }
    remote_session_join_stream(session);
    session->prompt_len = 0;
    session->output_len = 0;
    session->output_pos = 0;
    session->last_batch_outputs = 0;
    session->stream_err = ASTRAL_OK;
    session->stream_started = false;
    session->stream_done = false;
    return ASTRAL_OK;
}

AstralErr remote_session_push_prompt_token(RemoteSession* session, int32_t token) {
    if (token >= 0 && token < 256) {
        if (session->prompt_len >= sizeof(session->prompt)) {
            return ASTRAL_E_NOMEM;
        }
        session->prompt[session->prompt_len++] = static_cast<uint8_t>(token);
    }
    return ASTRAL_OK;
}

void remote_session_accept_token(RemoteSession* session, int32_t token) {
    if (token >= 0 && token < 256) {
        std::lock_guard<std::mutex> lock(session->stream_mutex);
        if (session->output_pos < session->output_len) {
            ++session->output_pos;
        }
    }
}

AstralErr remote_session_feed(void* session_ctx, const int32_t* tokens, uint32_t count) {
    RemoteSession* session = static_cast<RemoteSession*>(session_ctx);
    if (session == nullptr || (count != 0 && tokens == nullptr)) {
        return ASTRAL_E_INVALID;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const AstralErr err = remote_session_push_prompt_token(session, tokens[i]);
        if (err != ASTRAL_OK) {
            return err;
        }
    }
    return ASTRAL_OK;
}

AstralErr remote_session_logits(void* session_ctx, AstralBackendLogitsView* out_view) {
    RemoteSession* session = static_cast<RemoteSession*>(session_ctx);
    if (session == nullptr || out_view == nullptr) {
        return ASTRAL_E_INVALID;
    }
    AstralErr err = remote_session_start_stream(session);
    if (err != ASTRAL_OK) {
        return err;
    }

    uint32_t next = kRemoteTokenEos;
    {
        std::unique_lock<std::mutex> lock(session->stream_mutex);
        while (session->output_pos >= session->output_len && !session->stream_done) {
            session->stream_cv.wait(lock);
        }
        if (session->output_pos < session->output_len) {
            next = static_cast<uint32_t>(session->output[session->output_pos]);
        } else if (session->stream_err != ASTRAL_OK) {
            return session->stream_err;
        }
    }

    for (uint32_t i = 0; i < kRemoteVocabSize; ++i) {
        session->logits[i] = kRemoteSuppressedLogit;
    }
    session->logits[next] = kRemoteSelectedLogit;

    out_view->logits = session->logits;
    out_view->vocab_size = kRemoteVocabSize;
    return ASTRAL_OK;
}

AstralErr remote_session_accept(void* session_ctx, int32_t token) {
    RemoteSession* session = static_cast<RemoteSession*>(session_ctx);
    if (session == nullptr) {
        return ASTRAL_E_INVALID;
    }
    remote_session_accept_token(session, token);
    return ASTRAL_OK;
}

AstralErr remote_session_batch_eval(void* session_ctx,
                                    const AstralBackendBatchToken* tokens,
                                    uint32_t token_count,
                                    uint32_t* out_output_count) {
    RemoteSession* session = static_cast<RemoteSession*>(session_ctx);
    if (session == nullptr || out_output_count == nullptr || (token_count != 0 && tokens == nullptr)) {
        return ASTRAL_E_INVALID;
    }
    session->last_batch_outputs = 0;

    for (uint32_t i = 0; i < token_count; ++i) {
        const AstralBackendBatchToken& token = tokens[i];
        if (token.slot_id != 0) {
            return ASTRAL_E_UNSUPPORTED;
        }
        AstralErr err = ASTRAL_OK;
        if (session->stream_started) {
            remote_session_accept_token(session, token.token);
        } else {
            err = remote_session_push_prompt_token(session, token.token);
        }
        if (err != ASTRAL_OK) {
            return err;
        }
        if (token.want_logits != 0) {
            AstralBackendLogitsView view{};
            err = remote_session_logits(session, &view);
            if (err != ASTRAL_OK) {
                return err;
            }
            ++session->last_batch_outputs;
        }
    }

    *out_output_count = session->last_batch_outputs;
    return ASTRAL_OK;
}

AstralErr remote_session_batch_logits(void* session_ctx, uint32_t output_index, AstralBackendLogitsView* out_view) {
    RemoteSession* session = static_cast<RemoteSession*>(session_ctx);
    if (session == nullptr || out_view == nullptr) {
        return ASTRAL_E_INVALID;
    }
    if (output_index >= session->last_batch_outputs) {
        return ASTRAL_E_INVALID;
    }
    out_view->logits = session->logits;
    out_view->vocab_size = kRemoteVocabSize;
    return ASTRAL_OK;
}

AstralErr remote_session_slot_reset(void* session_ctx, uint32_t slot_id) {
    if (slot_id != 0) {
        return ASTRAL_E_UNSUPPORTED;
    }
    return remote_session_reset(session_ctx);
}

void* remote_embedder_create(void* model_ctx, AstralErr* out_err) {
    if (out_err == nullptr) {
        return nullptr;
    }
    RemoteModel* model = static_cast<RemoteModel*>(model_ctx);
    if (model == nullptr) {
        *out_err = ASTRAL_E_INVALID;
        return nullptr;
    }
    RemoteEmbedder* embedder = core::runtime_new<RemoteEmbedder>();
    if (embedder == nullptr) {
        *out_err = ASTRAL_E_NOMEM;
        return nullptr;
    }
    embedder->model = model;
    *out_err = ASTRAL_OK;
    return embedder;
}

void remote_embedder_destroy(void* embedder_ctx) {
    core::runtime_delete(static_cast<RemoteEmbedder*>(embedder_ctx));
}

AstralErr remote_embedder_reset(void* embedder_ctx) {
    return embedder_ctx != nullptr ? ASTRAL_OK : ASTRAL_E_INVALID;
}

AstralErr remote_embedder_embed(void* embedder_ctx, const int32_t* tokens, uint32_t count, float* out_vec, uint32_t vec_dim) {
    RemoteEmbedder* embedder = static_cast<RemoteEmbedder*>(embedder_ctx);
    if (embedder == nullptr || out_vec == nullptr || (count != 0 && tokens == nullptr)) {
        return ASTRAL_E_INVALID;
    }

    char text[kRemoteMaxPromptBytes];
    uint32_t text_len = 0;
    for (uint32_t i = 0; i < count && text_len < sizeof(text); ++i) {
        const int32_t token = tokens[i];
        if (token >= 0 && token < 256) {
            text[text_len++] = static_cast<char>(token);
        }
    }

    std::string body;
    const AstralErr err = post_remote(embedder->model, "/embeddings", text, text_len, &body);
    if (err != ASTRAL_OK) {
        return err;
    }
    for (uint32_t i = 0; i < vec_dim; ++i) {
        out_vec[i] = 0.0f;
    }
    const uint32_t parsed = parse_float_list(body.data(), static_cast<uint32_t>(body.size()), out_vec, vec_dim);
    return parsed != 0 ? ASTRAL_OK : ASTRAL_E_BACKEND;
}

const AstralBackendOps kRemoteOps = [] {
  AstralBackendOps ops{};
  ops.model_load = remote_model_load;
  ops.model_unload = remote_model_unload;
  ops.tokenize = remote_tokenize;
  ops.detokenize = remote_detokenize;
  ops.model_info = remote_model_info;
  ops.model_special_tokens = remote_model_special_tokens;
  ops.model_embedding_dim = remote_model_embedding_dim;
  ops.session_create = remote_session_create;
  ops.session_destroy = remote_session_destroy;
  ops.session_reset = remote_session_reset;
  ops.session_feed = remote_session_feed;
  ops.session_logits = remote_session_logits;
  ops.session_accept = remote_session_accept;
  ops.session_batch_eval = remote_session_batch_eval;
  ops.session_batch_logits = remote_session_batch_logits;
  ops.session_slot_reset = remote_session_slot_reset;
  ops.embedder_create = remote_embedder_create;
  ops.embedder_destroy = remote_embedder_destroy;
  ops.embedder_reset = remote_embedder_reset;
  ops.embedder_embed = remote_embedder_embed;
  return ops;
}();

const AstralBackendProvider kRemoteProvider = {
    /*name=*/"remote",
    /*ops=*/&kRemoteOps,
    /*supports_gpu=*/0,
    /*min_gpu_layers=*/0,
};

} // namespace

const BackendProvider* builtin_remote_backend_provider() {
    return &kRemoteProvider;
}

} // namespace astral::backend
