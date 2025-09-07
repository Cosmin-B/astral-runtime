/**
 * main.cpp - Test runner entry point
 */

#include "test_framework.hpp"

#if defined(__linux__)
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <execinfo.h>
#include <unistd.h>

namespace {

void crash_handler(int sig) {
    void* frames[128];
    const int n = ::backtrace(frames, static_cast<int>(sizeof(frames) / sizeof(frames[0])));

    const char* name = (sig == SIGSEGV) ? "SIGSEGV" :
                       (sig == SIGABRT) ? "SIGABRT" :
                       (sig == SIGILL) ? "SIGILL" :
                       (sig == SIGFPE) ? "SIGFPE" :
                       (sig == SIGBUS) ? "SIGBUS" : "SIGNAL";

    std::fprintf(stderr, "\n[CRASH] %s (%d)\n", name, sig);
    std::fprintf(stderr, "[CRASH] backtrace frames=%d\n", n);
    ::backtrace_symbols_fd(frames, n, STDERR_FILENO);
    std::fprintf(stderr, "\n");
    std::_Exit(128 + sig);
}

void install_crash_handlers() {
    struct sigaction sa {};
    sa.sa_handler = crash_handler;
    ::sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND;
    ::sigaction(SIGSEGV, &sa, nullptr);
    ::sigaction(SIGABRT, &sa, nullptr);
    ::sigaction(SIGILL, &sa, nullptr);
    ::sigaction(SIGFPE, &sa, nullptr);
    ::sigaction(SIGBUS, &sa, nullptr);
}

} // namespace
#endif

int main() {
#if defined(__linux__)
    install_crash_handlers();
#endif
    return astral::testing::TestRegistry::instance().run_all();
}
