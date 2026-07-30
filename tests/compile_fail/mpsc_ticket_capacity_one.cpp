#include "src/concurrency/mpsc_ticket_ring.hpp"

#include <cstdint>

astral::concurrency::MpscTicketRing<uint32_t, 1> g_queue;

int main() {
  return g_queue.empty() ? 0 : 1;
}
