#pragma once

#include <cstdint>

namespace astral::platform {

struct ReadOnlyFileMap {
  const uint8_t* data;
  uint64_t size;
  void* mapping;
  void* file;
};

bool file_map_readonly(const char* path, ReadOnlyFileMap* out_map);
void file_unmap_readonly(ReadOnlyFileMap* map);

} // namespace astral::platform
