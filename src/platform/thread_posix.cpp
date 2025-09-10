#if !defined(_WIN32)

#include "thread.h"

#include <unistd.h>

namespace astral::platform {

static void* thread_entry(void* arg) {
    Thread* t = static_cast<Thread*>(arg);
    if (t && t->fn) {
        t->fn(t->user);
    }
    return nullptr;
}

uint32_t hardware_concurrency() {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n > 0) {
        return static_cast<uint32_t>(n);
    }
    return 0;
}

bool thread_start(Thread* t, ThreadFn fn, void* user) {
    if (!t || !fn) {
        return false;
    }

    t->fn = fn;
    t->user = user;
    t->started = false;

    const int rc = pthread_create(&t->handle, nullptr, &thread_entry, t);
    if (rc != 0) {
        return false;
    }

    t->started = true;
    return true;
}

void thread_join(Thread* t) {
    if (!t || !t->started) {
        return;
    }

    (void)pthread_join(t->handle, nullptr);
    t->started = false;
}

} // namespace astral::platform

#endif

