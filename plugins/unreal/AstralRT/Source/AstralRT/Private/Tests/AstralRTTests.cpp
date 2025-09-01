#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"

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
    if (err != ASTRAL_OK) {
        astral_session_destroy(session);
        astral_model_release(model);
        return false;
    }

    TArray<uint8> out2;
    const bool ok2 = run_once(out2);
    TestTrue(TEXT("second decode ok"), ok2);
    TestTrue(TEXT("mock output repeats"), out2 == out1);

    astral_session_destroy(session);
    astral_model_release(model);
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS

