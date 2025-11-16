#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"

#include "AstralEmbedder.h"
#include "AstralMediaLibrary.h"
#include "AstralModel.h"
#include "AstralSession.h"
#include "IAstralRT.h"
#include "astral_rt.h"

namespace {

static bool ensure_astral_initialized(FAutomationTestBase& test) {
    // Force-load module if needed.
    (void)IAstralRT::Get();

    if (!IAstralRT::IsAvailable() || !IAstralRT::Get().IsInitialized()) {
        test.AddError(TEXT("AstralRT runtime is not initialized."));
        return false;
    }
    return true;
}

static bool bytes_equal_ascii(const TArray<uint8>& bytes, const char* lit) {
    if (lit == nullptr) {
        return false;
    }
    const int32 lit_len = FCStringAnsi::Strlen(lit);
    if (bytes.Num() != lit_len) {
        return false;
    }
    return FMemory::Memcmp(bytes.GetData(), lit, lit_len) == 0;
}

static FString bytes_to_ascii_string(const TArray<uint8>& bytes) {
    TArray<ANSICHAR> text;
    text.Reserve(bytes.Num() + 1);
    for (const uint8 b : bytes) {
        text.Add(static_cast<ANSICHAR>(b));
    }
    text.Add('\0');
    return FString(ANSI_TO_TCHAR(text.GetData()));
}

static FString bytes_to_hex_string(const TArray<uint8>& bytes) {
    FString out;
    out.Reserve(bytes.Num() * 3);
    for (const uint8 b : bytes) {
        out += FString::Printf(TEXT("%02x "), static_cast<uint32>(b));
    }
    return out;
}

static void append_ascii(TArray<uint8>& out, const char* lit) {
    out.Reset();
    if (lit == nullptr) {
        return;
    }
    const int32 lit_len = FCStringAnsi::Strlen(lit);
    out.Append(reinterpret_cast<const uint8*>(lit), lit_len);
}

static bool run_mock_session_once(AstralHandle session, TArray<uint8>& out_bytes) {
    out_bytes.Reset();
    out_bytes.Reserve(32);

    const char* prompt = "hi";
    AstralSpanU8 prompt_span{};
    prompt_span.data = reinterpret_cast<const uint8_t*>(prompt);
    prompt_span.len = static_cast<uint32_t>(FCStringAnsi::Strlen(prompt));

    AstralErr e = astral_session_feed(session, prompt_span, 1);
    if (e != ASTRAL_OK) {
        return false;
    }

    e = astral_session_decode(session);
    if (e != ASTRAL_OK) {
        return false;
    }

    e = astral_session_wait(session, 5000);
    if (e != ASTRAL_OK) {
        return false;
    }

    uint8_t buf[64];
    for (;;) {
        AstralMutSpanU8 out{};
        out.data = buf;
        out.len = sizeof(buf);

        const int32_t n = astral_stream_read(session, out, 0);
        if (n == ASTRAL_E_TIMEOUT) {
            continue;
        }
        if (n < 0) {
            return false;
        }
        if (n == 0) {
            break;
        }
        out_bytes.Append(buf, n);
    }
    return true;
}

} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTModuleInitTest,
    "AstralRT.Module.Init",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTModuleInitTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    const bool ok = ensure_astral_initialized(*this);
    if (!ok) {
        return false;
    }
    TestTrue(TEXT("AstralRT initialized"), ok);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTModuleShutdownRestartTest,
    "AstralRT.Module.ShutdownRestart",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTModuleShutdownRestartTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    IAstralRT& Runtime = IAstralRT::Get();
    Runtime.ShutdownModule();
    TestFalse(TEXT("runtime reports uninitialized after shutdown"), Runtime.IsInitialized());

    Runtime.ShutdownModule();
    TestFalse(TEXT("second shutdown remains uninitialized"), Runtime.IsInitialized());

    Runtime.StartupModule();
    TestTrue(TEXT("runtime reinitializes after startup"), Runtime.IsInitialized());
    if (!Runtime.IsInitialized()) {
        return false;
    }

    AstralModelDesc model_desc{};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* backend = "mock";
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(FCStringAnsi::Strlen(backend));
    model_desc.n_ctx = 128;

    AstralHandle model = 0;
    const AstralErr err = astral_model_load(&model_desc, &model);
    TestEqual(TEXT("mock model loads after module restart"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    TestTrue(TEXT("mock model handle after module restart"), model != 0);
    if (model != 0) {
        astral_model_release(model);
    }
    return err == ASTRAL_OK && model != 0;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTModuleEnginePreExitTest,
    "AstralRT.Module.EnginePreExit",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTModuleEnginePreExitTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    IAstralRT& Runtime = IAstralRT::Get();
    Runtime.SimulateEnginePreExitForAutomation();
    TestFalse(TEXT("runtime reports uninitialized after engine pre-exit"), Runtime.IsInitialized());

    Runtime.SimulateEnginePreExitForAutomation();
    TestFalse(TEXT("second engine pre-exit remains uninitialized"), Runtime.IsInitialized());

    Runtime.StartupModule();
    TestTrue(TEXT("runtime reinitializes after engine pre-exit"), Runtime.IsInitialized());
    return Runtime.IsInitialized();
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTFMemoryAllocatorTest,
    "AstralRT.Memory.FMemoryAllocator",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTFMemoryAllocatorTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    IAstralRT& Runtime = IAstralRT::Get();
    Runtime.ResetAllocatorStats();

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ContextSize = 128;
    bool ok = Model->Load(ModelDesc);
    TestTrue(TEXT("model load uses FMemory allocator"), ok);
    if (!ok) {
        return false;
    }

    UAstralSession* Session = NewObject<UAstralSession>();
    TestNotNull(TEXT("session allocated"), Session);

    FAstralSessionDesc SessionDesc{};
    SessionDesc.MaxTokens = 16;
    SessionDesc.Temperature = 0.0f;
    SessionDesc.TopK = 0;
    SessionDesc.TopP = 1.0f;
    SessionDesc.bStreamEnabled = false;
    SessionDesc.Seed = 7;

    ok = Session->Create(Model, SessionDesc);
    TestTrue(TEXT("session create uses FMemory allocator"), ok);
    if (!ok) {
        Model->Release();
        Model->ConditionalBeginDestroy();
        return false;
    }

    const FAstralRTAllocatorStats LiveStats = Runtime.GetAllocatorStats();
    TestTrue(TEXT("native alloc callback called"), LiveStats.AllocCalls > 0);
    TestTrue(TEXT("native alloc callback tracked bytes"), LiveStats.AllocBytes > 0);

    Session->ConditionalBeginDestroy();
    Model->Release();
    Model->ConditionalBeginDestroy();

    const FAstralRTAllocatorStats ReleasedStats = Runtime.GetAllocatorStats();
    TestTrue(TEXT("native free callback called"), ReleasedStats.FreeCalls > 0);
    TestTrue(TEXT("native free callback tracked bytes"), ReleasedStats.FreeBytes > 0);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMockE2ETest,
    "AstralRT.Mock.E2E",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMockE2ETest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    AstralModelDesc model_desc{};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* backend = "mock";
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(FCStringAnsi::Strlen(backend));
    model_desc.n_ctx = 128;

    AstralHandle model = 0;
    AstralErr err = astral_model_load(&model_desc, &model);
    TestEqual(TEXT("astral_model_load"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    if (err != ASTRAL_OK || model == 0) {
        return false;
    }

    AstralCaps caps = 0;
    err = astral_model_caps(model, &caps);
    TestEqual(TEXT("astral_model_caps"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    TestTrue(TEXT("mock caps include stop seqs"), (caps & ASTRAL_CAP_STOP_SEQS) != 0);
    TestTrue(TEXT("mock caps include sampler ext"), (caps & ASTRAL_CAP_SAMPLER_EXT) != 0);

    AstralSessionDesc session_desc{};
    session_desc.model = model;
    session_desc.max_tokens = 32;
    session_desc.temperature = 0.0f;
    session_desc.top_k = 0;
    session_desc.top_p = 1.0f;
    session_desc.stream_enabled = 1;
    session_desc.seed = 1;

    AstralHandle session = 0;
    err = astral_session_create(&session_desc, &session);
    TestEqual(TEXT("astral_session_create"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    if (err != ASTRAL_OK || session == 0) {
        astral_model_release(model);
        return false;
    }

    TArray<uint8> out1;
    const bool ok1 = run_mock_session_once(session, out1);
    TestTrue(TEXT("first decode ok"), ok1);
    TestTrue(TEXT("mock output == mock-backend"), bytes_equal_ascii(out1, "mock-backend"));

    err = astral_session_reset(session, &session_desc);
    TestEqual(TEXT("astral_session_reset"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));

    err = astral_session_stop_clear(session);
    TestEqual(TEXT("astral_session_stop_clear"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    const char* stop = "backend";
    AstralSpanU8 stop_span{};
    stop_span.data = reinterpret_cast<const uint8_t*>(stop);
    stop_span.len = static_cast<uint32_t>(FCStringAnsi::Strlen(stop));
    TestEqual(TEXT("stop span len"), static_cast<int32>(stop_span.len), 7);

    int32_t stop_tokens[16];
    uint32_t stop_token_count = 0;
    err = astral_tokenize(model, stop_span, stop_tokens, 16, 0, 0, &stop_token_count);
    TestEqual(TEXT("stop tokenize"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    TestEqual(TEXT("stop token count"), static_cast<int32>(stop_token_count), 7);
    if (stop_token_count > 0) {
        TestEqual(TEXT("stop token first"), stop_tokens[0], static_cast<int32>('b'));
        TestEqual(TEXT("stop token last"), stop_tokens[stop_token_count - 1], static_cast<int32>('d'));
    }

    err = astral_session_stop_add_utf8(session, stop_span);
    TestEqual(TEXT("astral_session_stop_add_utf8"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));

    TArray<uint8> out2;
    const bool ok2 = run_mock_session_once(session, out2);
    TestTrue(TEXT("second decode ok"), ok2);
    TestEqual(TEXT("stop output byte count"), out2.Num(), 5);
    TestEqual(TEXT("stop suppresses backend suffix"), bytes_to_ascii_string(out2), FString(TEXT("mock-")));
    if (!bytes_equal_ascii(out2, "mock-")) {
        AddError(FString::Printf(TEXT("stop output hex: %s"), *bytes_to_hex_string(out2)));
    }

    astral_session_destroy(session);
    astral_model_release(model);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMockEmbeddingsTest,
    "AstralRT.Mock.Embeddings",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMockEmbeddingsTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    AstralModelDesc model_desc{};
    model_desc.size = sizeof(AstralModelDesc);
    model_desc.source_kind = ASTRAL_MODEL_SOURCE_PATH;
    const char* backend = "mock";
    model_desc.backend_name.data = reinterpret_cast<const uint8_t*>(backend);
    model_desc.backend_name.len = static_cast<uint32_t>(FCStringAnsi::Strlen(backend));
    model_desc.n_ctx = 128;
    model_desc.embeddings_only = 1;

    AstralHandle model = 0;
    AstralErr err = astral_model_load(&model_desc, &model);
    TestEqual(TEXT("astral_model_load"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    if (err != ASTRAL_OK || model == 0) {
        return false;
    }

    uint32_t dim = 0;
    err = astral_model_embedding_dim(model, &dim);
    TestEqual(TEXT("astral_model_embedding_dim"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    TestEqual(TEXT("mock dim"), static_cast<int32>(dim), 8);

    AstralHandle emb = 0;
    err = astral_embed_create(model, &emb);
    TestEqual(TEXT("astral_embed_create"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    if (err != ASTRAL_OK || emb == 0) {
        astral_model_release(model);
        return false;
    }

    const char* text = "abc";
    AstralSpanU8 text_span{};
    text_span.data = reinterpret_cast<const uint8_t*>(text);
    text_span.len = static_cast<uint32_t>(FCStringAnsi::Strlen(text));

    uint64_t ticket = 0;
    err = astral_embed_enqueue(emb, text_span, &ticket);
    TestEqual(TEXT("astral_embed_enqueue"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));
    if (err != ASTRAL_OK) {
        astral_embed_destroy(emb);
        astral_model_release(model);
        return false;
    }

    TArray<float> vec;
    vec.SetNumUninitialized(static_cast<int32>(dim));

    AstralMutSpanU8 out{};
    out.data = reinterpret_cast<uint8_t*>(vec.GetData());
    out.len = static_cast<uint32_t>(vec.Num() * sizeof(float));

    err = astral_embed_collect(emb, ticket, out);
    TestEqual(TEXT("astral_embed_collect"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));

    // mock embedder: sum(tokens incl BOS=256) + i, so for "abc": 256 + 97 + 98 + 99 = 550.
    TestEqual(TEXT("vec[0]"), vec[0], 550.0f);
    TestEqual(TEXT("vec[1]"), vec[1], 551.0f);
    TestEqual(TEXT("vec[7]"), vec[7], 557.0f);

    astral_embed_destroy(emb);
    astral_model_release(model);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMockEmbedderQueuePressureTest,
    "AstralRT.Mock.EmbedderQueuePressure",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMockEmbedderQueuePressureTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ContextSize = 128;
    ModelDesc.bEmbeddingsOnly = true;
    bool ok = Model->Load(ModelDesc);
    TestTrue(TEXT("model load"), ok);
    if (!ok) {
        return false;
    }

    UAstralEmbedder* Embedder = NewObject<UAstralEmbedder>();
    TestNotNull(TEXT("embedder allocated"), Embedder);

    ok = Embedder->Create(Model);
    TestTrue(TEXT("embedder create"), ok);
    if (!ok) {
        Model->Release();
        return false;
    }

    constexpr int32 Inflight = 8;
    const char* Texts[Inflight] = {
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
    };
    int64 Tickets[Inflight] = {};
    TArray<uint8> Bytes;

    for (int32 i = 0; i < Inflight; ++i) {
        append_ascii(Bytes, Texts[i]);
        ok = Embedder->EnqueueUtf8Bytes(Bytes, Tickets[i]);
        TestTrue(TEXT("enqueue inflight ticket"), ok);
        TestTrue(TEXT("ticket valid"), Tickets[i] > 0);
    }

    int64 OverflowTicket = 0;
    append_ascii(Bytes, "overflow");
    ok = Embedder->EnqueueUtf8Bytes(Bytes, OverflowTicket);
    TestFalse(TEXT("overflow returns busy through wrapper"), ok);
    TestEqual(TEXT("overflow ticket remains zero"), OverflowTicket, static_cast<int64>(0));

    TArray<float> Vec;
    for (int32 Offset = 0; Offset < Inflight; ++Offset) {
        const int32 Index = Inflight - 1 - Offset;
        ok = Embedder->Collect(Tickets[Index], Vec);
        TestTrue(TEXT("collect out of order"), ok);
        TestEqual(TEXT("vector size"), Vec.Num(), Embedder->GetDim());
    }

    ok = Embedder->Collect(Tickets[0], Vec);
    TestFalse(TEXT("stale ticket rejected"), ok);

    int64 ReuseTicket = 0;
    append_ascii(Bytes, "reuse");
    ok = Embedder->EnqueueUtf8Bytes(Bytes, ReuseTicket);
    TestTrue(TEXT("enqueue after drain"), ok);
    TestTrue(TEXT("reuse ticket valid"), ReuseTicket > 0);
    ok = Embedder->Collect(ReuseTicket, Vec);
    TestTrue(TEXT("collect reuse ticket"), ok);

    Embedder->Destroy();
    Embedder->ConditionalBeginDestroy();
    Model->Release();
    Model->ConditionalBeginDestroy();
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMockMemoryModelSourceTest,
    "AstralRT.Mock.MemoryModelSource",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMockMemoryModelSourceTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.SourceKind = EAstralModelSourceKind::Memory;
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ContextSize = 128;
    ModelDesc.ModelBytes.SetNumUninitialized(4);
    ModelDesc.ModelBytes[0] = 'm';
    ModelDesc.ModelBytes[1] = 'o';
    ModelDesc.ModelBytes[2] = 'c';
    ModelDesc.ModelBytes[3] = 'k';

    const bool ok = Model->Load(ModelDesc);
    TestTrue(TEXT("memory source model load"), ok);
    TestTrue(TEXT("model valid after memory load"), Model->IsValid());

    Model->Release();
    Model->ConditionalBeginDestroy();
    return ok;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMockFailedLoadRecoveryTest,
    "AstralRT.Mock.FailedLoadRecovery",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMockFailedLoadRecoveryTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.SourceKind = EAstralModelSourceKind::Memory;
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ContextSize = 128;

    AddExpectedError(TEXT("AstralRT: memory model source has no bytes"), EAutomationExpectedErrorFlags::Contains, 1);
    bool ok = Model->Load(ModelDesc);
    TestFalse(TEXT("empty memory model load fails"), ok);
    TestFalse(TEXT("failed load leaves model invalid"), Model->IsValid());
    TestEqual(TEXT("failed load leaves handle zero"), Model->GetHandle(), static_cast<uint64>(0));

    int32 Dim = 7;
    ok = Model->GetEmbeddingDim(Dim);
    TestFalse(TEXT("failed load has no embedding dim"), ok);
    TestEqual(TEXT("failed dim query zeroes output"), Dim, 0);

    UAstralSession* Session = NewObject<UAstralSession>();
    TestNotNull(TEXT("session allocated"), Session);

    FAstralSessionDesc SessionDesc{};
    SessionDesc.MaxTokens = 16;
    SessionDesc.Temperature = 0.0f;
    SessionDesc.TopK = 0;
    SessionDesc.TopP = 1.0f;
    SessionDesc.bStreamEnabled = false;
    SessionDesc.Seed = 11;

    AddExpectedError(TEXT("AstralRT: invalid model"), EAutomationExpectedErrorFlags::Contains, 1);
    ok = Session->Create(Model, SessionDesc);
    TestFalse(TEXT("invalid model create rejected"), ok);
    TestFalse(TEXT("failed create leaves session invalid"), Session->IsValid());

    ModelDesc.ModelBytes.SetNumUninitialized(4);
    ModelDesc.ModelBytes[0] = 'm';
    ModelDesc.ModelBytes[1] = 'o';
    ModelDesc.ModelBytes[2] = 'c';
    ModelDesc.ModelBytes[3] = 'k';

    ok = Model->Load(ModelDesc);
    TestTrue(TEXT("model recovers after failed load"), ok);
    TestTrue(TEXT("recovered model valid"), Model->IsValid());

    ok = Session->Create(Model, SessionDesc);
    TestTrue(TEXT("session create after model recovery"), ok);
    TestTrue(TEXT("recovered session valid"), Session->IsValid());

    Session->ConditionalBeginDestroy();
    Model->Release();
    Model->Release();
    TestFalse(TEXT("double release leaves model invalid"), Model->IsValid());
    Model->ConditionalBeginDestroy();
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMockSessionCancelResetTest,
    "AstralRT.Mock.SessionCancelReset",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMockSessionCancelResetTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ModelPath = TEXT("infinite");
    ModelDesc.ContextSize = 128;
    bool ok = Model->Load(ModelDesc);
    TestTrue(TEXT("infinite mock model load"), ok);
    if (!ok) {
        return false;
    }

    UAstralSession* Session = NewObject<UAstralSession>();
    TestNotNull(TEXT("session allocated"), Session);

    FAstralSessionDesc SessionDesc{};
    SessionDesc.MaxTokens = 256;
    SessionDesc.Temperature = 0.0f;
    SessionDesc.TopK = 0;
    SessionDesc.TopP = 1.0f;
    SessionDesc.bStreamEnabled = true;
    SessionDesc.Seed = 7;

    ok = Session->Create(Model, SessionDesc);
    TestTrue(TEXT("session create"), ok);
    if (!ok) {
        Model->Release();
        return false;
    }

    ok = Session->FeedPrompt(TEXT("hi"), true);
    TestTrue(TEXT("first feed"), ok);
    ok = Session->Decode();
    TestTrue(TEXT("first decode"), ok);
    ok = Session->Cancel();
    TestTrue(TEXT("first cancel"), ok);
    TestEqual(TEXT("first wait canceled"), Session->Wait(5000), static_cast<int32>(ASTRAL_E_CANCELED));

    SessionDesc.bStreamEnabled = false;
    ok = Session->Reset(SessionDesc);
    TestTrue(TEXT("reset after cancel"), ok);

    ok = Session->FeedPrompt(TEXT("reuse"), true);
    TestTrue(TEXT("second feed"), ok);
    ok = Session->Decode();
    TestTrue(TEXT("second decode"), ok);
    ok = Session->Cancel();
    TestTrue(TEXT("second cancel"), ok);
    TestEqual(TEXT("second wait canceled"), Session->Wait(5000), static_cast<int32>(ASTRAL_E_CANCELED));

    Session->ConditionalBeginDestroy();
    Model->Release();
    Model->ConditionalBeginDestroy();
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMockDestroyInvalidationTest,
    "AstralRT.Mock.DestroyInvalidation",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMockDestroyInvalidationTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ModelPath = TEXT("infinite");
    ModelDesc.ContextSize = 128;
    bool ok = Model->Load(ModelDesc);
    TestTrue(TEXT("infinite mock model load"), ok);
    if (!ok) {
        return false;
    }

    UAstralSession* Session = NewObject<UAstralSession>();
    TestNotNull(TEXT("session allocated"), Session);

    FAstralSessionDesc SessionDesc{};
    SessionDesc.MaxTokens = 256;
    SessionDesc.Temperature = 0.0f;
    SessionDesc.TopK = 0;
    SessionDesc.TopP = 1.0f;
    SessionDesc.bStreamEnabled = true;
    SessionDesc.Seed = 17;

    ok = Session->Create(Model, SessionDesc);
    TestTrue(TEXT("streaming session create"), ok);
    if (!ok) {
        Model->Release();
        return false;
    }

    ok = Session->FeedPrompt(TEXT("hi"), true);
    TestTrue(TEXT("feed before destroy"), ok);
    ok = Session->Decode();
    TestTrue(TEXT("decode before destroy"), ok);
    ok = Session->Cancel();
    TestTrue(TEXT("cancel before destroy"), ok);
    TestEqual(TEXT("wait canceled before destroy"), Session->Wait(5000), static_cast<int32>(ASTRAL_E_CANCELED));

    Session->ConditionalBeginDestroy();
    TestFalse(TEXT("destroy clears session valid state"), Session->IsValid());
    TestEqual(TEXT("wait after destroy is invalid"), Session->Wait(0), static_cast<int32>(ASTRAL_E_INVALID));
    TestFalse(TEXT("cancel after destroy fails"), Session->Cancel());
    TestFalse(TEXT("reset after destroy fails"), Session->Reset(SessionDesc));

    TArray<uint8> Out;
    Out.SetNumUninitialized(16);
    TestEqual(TEXT("stream read after destroy is invalid"), Session->StreamRead(Out, 0), static_cast<int32>(ASTRAL_E_INVALID));

    Model->Release();
    TestFalse(TEXT("model release clears valid state"), Model->IsValid());
    Model->Release();
    TestFalse(TEXT("second model release stays invalid"), Model->IsValid());
    Model->ConditionalBeginDestroy();
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMockMediaFeedTest,
    "AstralRT.Mock.MediaFeed",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMockMediaFeedTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ContextSize = 128;
    bool ok = Model->Load(ModelDesc);
    TestTrue(TEXT("model load"), ok);
    if (!ok) {
        return false;
    }

    UAstralSession* Session = NewObject<UAstralSession>();
    TestNotNull(TEXT("session allocated"), Session);

    FAstralSessionDesc SessionDesc{};
    SessionDesc.MaxTokens = 16;
    SessionDesc.Temperature = 0.0f;
    SessionDesc.TopK = 0;
    SessionDesc.TopP = 1.0f;
    SessionDesc.bStreamEnabled = false;
    SessionDesc.Seed = 42;

    ok = Session->Create(Model, SessionDesc);
    TestTrue(TEXT("session create"), ok);
    if (!ok) {
        Model->Release();
        return false;
    }

    FAstralImageDesc Image{};
    Image.Format = EAstralImageFormat::RGB8;
    Image.Width = 1;
    Image.Height = 1;
    Image.Pixels.SetNumZeroed(3);
    ok = Session->FeedImage(Image, true);
    TestTrue(TEXT("feed image"), ok);

    FAstralAudioDesc Audio{};
    Audio.Format = EAstralAudioFormat::I16;
    Audio.Channels = 1;
    Audio.SampleRate = 16000;
    Audio.FrameCount = 4;
    Audio.Samples.SetNumZeroed(8);
    ok = Session->FeedAudio(Audio, true);
    TestTrue(TEXT("feed audio"), ok);

    Session->ConditionalBeginDestroy();
    Model->Release();
    Model->ConditionalBeginDestroy();
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMediaDescriptorHelpersTest,
    "AstralRT.Media.DescriptorHelpers",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMediaDescriptorHelpersTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    TArray<uint8> Rgba;
    Rgba.SetNumZeroed(2 * 2 * 4);

    FAstralImageDesc Image{};
    bool ok = UAstralMediaLibrary::MakeRGBA8ImageFromBytes(Rgba, 2, 2, Image);
    TestTrue(TEXT("RGBA bytes descriptor"), ok);
    TestEqual(TEXT("image width"), static_cast<int32>(Image.Width), 2);
    TestEqual(TEXT("image height"), static_cast<int32>(Image.Height), 2);
    TestEqual(TEXT("image stride"), static_cast<int32>(Image.RowStride), 8);
    TestEqual(TEXT("image byte count"), Image.Pixels.Num(), Rgba.Num());
    const FAstralImageDesc GoodImage = Image;

    TArray<uint8> BadRgba;
    BadRgba.SetNumZeroed(3);
    ok = UAstralMediaLibrary::MakeRGBA8ImageFromBytes(BadRgba, 2, 2, Image);
    TestFalse(TEXT("reject undersized RGBA bytes"), ok);

    TArray<uint8> Pcm;
    Pcm.SetNumZeroed(4 * static_cast<int32>(sizeof(int16)));

    FAstralAudioDesc Audio{};
    ok = UAstralMediaLibrary::MakePCM16AudioFromBytes(Pcm, 1, 16000, Audio);
    TestTrue(TEXT("PCM16 descriptor"), ok);
    TestEqual(TEXT("audio format"), static_cast<uint32>(Audio.Format), static_cast<uint32>(EAstralAudioFormat::I16));
    TestEqual(TEXT("audio channels"), static_cast<int32>(Audio.Channels), 1);
    TestEqual(TEXT("audio sample rate"), static_cast<int32>(Audio.SampleRate), 16000);
    TestEqual(TEXT("audio frame count"), static_cast<int32>(Audio.FrameCount), 4);
    const FAstralAudioDesc GoodAudio = Audio;

    TArray<uint8> BadPcm;
    BadPcm.SetNumZeroed(3);
    ok = UAstralMediaLibrary::MakePCM16AudioFromBytes(BadPcm, 2, 16000, Audio);
    TestFalse(TEXT("reject unaligned PCM16 bytes"), ok);

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ContextSize = 128;
    ok = Model->Load(ModelDesc);
    TestTrue(TEXT("model load"), ok);
    if (!ok) {
        return false;
    }

    UAstralSession* Session = NewObject<UAstralSession>();
    TestNotNull(TEXT("session allocated"), Session);

    FAstralSessionDesc SessionDesc{};
    SessionDesc.MaxTokens = 16;
    SessionDesc.Temperature = 0.0f;
    SessionDesc.TopK = 0;
    SessionDesc.TopP = 1.0f;
    SessionDesc.bStreamEnabled = false;
    SessionDesc.Seed = 13;

    ok = Session->Create(Model, SessionDesc);
    TestTrue(TEXT("session create"), ok);
    if (!ok) {
        Model->Release();
        return false;
    }

    ok = Session->FeedImage(GoodImage, false);
    TestTrue(TEXT("feed helper image"), ok);
    ok = Session->FeedAudio(GoodAudio, true);
    TestTrue(TEXT("feed helper audio"), ok);

    Session->ConditionalBeginDestroy();
    Model->Release();
    Model->ConditionalBeginDestroy();
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FAstralRTMockMultimodalEmbedTest,
    "AstralRT.Mock.MultimodalEmbed",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMockMultimodalEmbedTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return false;
    }

    UAstralModel* Model = NewObject<UAstralModel>();
    TestNotNull(TEXT("model allocated"), Model);

    FAstralModelDesc ModelDesc{};
    ModelDesc.BackendName = TEXT("mock");
    ModelDesc.ContextSize = 128;
    ModelDesc.bEmbeddingsOnly = true;
    bool ok = Model->Load(ModelDesc);
    TestTrue(TEXT("model load"), ok);
    if (!ok) {
        return false;
    }

    UAstralEmbedder* Embedder = NewObject<UAstralEmbedder>();
    TestNotNull(TEXT("embedder allocated"), Embedder);

    ok = Embedder->Create(Model);
    TestTrue(TEXT("embedder create"), ok);
    if (!ok) {
        Model->Release();
        return false;
    }

    FAstralImageDesc Image{};
    Image.Format = EAstralImageFormat::RGB8;
    Image.Width = 1;
    Image.Height = 1;
    Image.RowStride = 3;
    Image.Pixels.SetNumZeroed(3);

    FAstralAudioDesc Audio{};
    Audio.Format = EAstralAudioFormat::I16;
    Audio.Channels = 1;
    Audio.SampleRate = 16000;
    Audio.FrameCount = 4;
    Audio.Samples.SetNumZeroed(8);

    int64 Ticket = 0;
    ok = Embedder->EnqueueMultimodal(TEXT("abc"), Image, Audio, true, false, Ticket);
    TestTrue(TEXT("enqueue multimodal"), ok);
    TestTrue(TEXT("ticket valid"), Ticket > 0);

    TArray<float> Vec;
    ok = Embedder->Collect(Ticket, Vec);
    TestTrue(TEXT("collect embedding"), ok);
    TestTrue(TEXT("vector size"), Vec.Num() == Embedder->GetDim());

    Embedder->Destroy();
    Embedder->ConditionalBeginDestroy();
    Model->Release();
    Model->ConditionalBeginDestroy();
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
