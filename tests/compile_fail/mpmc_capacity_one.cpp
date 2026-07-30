#include "src/concurrency/mpmc_queue.hpp"

#include <cstdint>

astral::concurrency::MpmcQueue<uint32_t, 1> g_queue;

int main() {
  return g_queue.empty() ? 0 : 1;
}
