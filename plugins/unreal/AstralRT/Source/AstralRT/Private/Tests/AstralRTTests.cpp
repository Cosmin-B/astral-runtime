#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"

#include "AstralEmbedder.h"
#include "AstralModel.h"
#include "AstralSession.h"
#include "IAstralRT.h"
#include "astral_rt.h"

namespace {

static bool ensure_astral_initialized(FAutomationTestBase& test) {
    // Force-load module if needed.
    (void)IAstralRT::Get();

    if (!IAstralRT::IsAvailable() || !IAstralRT::Get().IsInitialized()) {
        test.AddWarning(TEXT("AstralRT runtime is not initialized; skipping test."));
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
        return true;
    }
    TestTrue(TEXT("AstralRT initialized"), ok);
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
        return true;
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

    auto run_once = [&](TArray<uint8>& out_bytes) -> bool {
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

        uint8_t buf[64];
        for (;;) {
            AstralMutSpanU8 out{};
            out.data = buf;
            out.len = sizeof(buf);

            const int32_t n = astral_stream_read(session, out, 1000);
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

        e = astral_session_wait(session, 5000);
        return e == ASTRAL_OK;
    };

    TArray<uint8> out1;
    const bool ok1 = run_once(out1);
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
    err = astral_session_stop_add_utf8(session, stop_span);
    TestEqual(TEXT("astral_session_stop_add_utf8"), static_cast<int32>(err), static_cast<int32>(ASTRAL_OK));

    TArray<uint8> out2;
    const bool ok2 = run_once(out2);
    TestTrue(TEXT("second decode ok"), ok2);
    TestTrue(TEXT("stop suppresses backend suffix"), bytes_equal_ascii(out2, "mock-"));

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
        return true;
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
    FAstralRTMockMediaFeedTest,
    "AstralRT.Mock.MediaFeed",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter
)

bool FAstralRTMockMediaFeedTest::RunTest(const FString& Parameters) {
    (void)Parameters;
    if (!ensure_astral_initialized(*this)) {
        return true;
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

    Session->BeginDestroy();
    Model->Release();
    Model->BeginDestroy();
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
        return true;
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
    Image.Pixels.SetNumZeroed(3);

    FAstralAudioDesc Audio{};
    Audio.Format = EAstralAudioFormat::I16;
    Audio.Channels = 1;
    Audio.SampleRate = 16000;
    Audio.FrameCount = 4;
    Audio.Samples.SetNumZeroed(8);

    int64 Ticket = 0;
    ok = Embedder->EnqueueMultimodal(TEXT("abc"), Image, Audio, true, true, Ticket);
    TestTrue(TEXT("enqueue multimodal"), ok);
    TestTrue(TEXT("ticket valid"), Ticket > 0);

    TArray<float> Vec;
    ok = Embedder->Collect(Ticket, Vec);
    TestTrue(TEXT("collect embedding"), ok);
    TestTrue(TEXT("vector size"), Vec.Num() == Embedder->GetDim());

    Embedder->Destroy();
    Embedder->BeginDestroy();
    Model->Release();
    Model->BeginDestroy();
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
