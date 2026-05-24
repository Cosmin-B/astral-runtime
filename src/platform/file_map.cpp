#include "file_map.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace astral::platform {

#if defined(_WIN32)

bool file_map_readonly(const char* path, ReadOnlyFileMap* out_map) {
  if (path == nullptr || out_map == nullptr) {
    return false;
  }
  *out_map = {};

  HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    return false;
  }

  LARGE_INTEGER size{};
  if (!GetFileSizeEx(file, &size) || size.QuadPart <= 0) {
    CloseHandle(file);
    return false;
  }

  HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (mapping == nullptr) {
    CloseHandle(file);
    return false;
  }

  void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  if (view == nullptr) {
    CloseHandle(mapping);
    CloseHandle(file);
    return false;
  }

  out_map->data = static_cast<const uint8_t*>(view);
  out_map->size = static_cast<uint64_t>(size.QuadPart);
  out_map->mapping = mapping;
  out_map->file = file;
  return true;
}

void file_unmap_readonly(ReadOnlyFileMap* map) {
  if (map == nullptr) {
    return;
  }
  if (map->data != nullptr) {
    UnmapViewOfFile(map->data);
  }
  if (map->mapping != nullptr) {
    CloseHandle(static_cast<HANDLE>(map->mapping));
  }
  if (map->file != nullptr) {
    CloseHandle(static_cast<HANDLE>(map->file));
  }
  *map = {};
}

#else

bool file_map_readonly(const char* path, ReadOnlyFileMap* out_map) {
  if (path == nullptr || out_map == nullptr) {
    return false;
  }
  *out_map = {};

  const int fd = open(path, O_RDONLY);
  if (fd < 0) {
    return false;
  }

  struct stat st{};
  if (fstat(fd, &st) != 0 || st.st_size <= 0) {
    close(fd);
    return false;
  }

  void* mapped = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (mapped == MAP_FAILED) {
    return false;
  }

  out_map->data = static_cast<const uint8_t*>(mapped);
  out_map->size = static_cast<uint64_t>(st.st_size);
  return true;
}

void file_unmap_readonly(ReadOnlyFileMap* map) {
  if (map == nullptr) {
    return;
  }
  if (map->data != nullptr && map->size != 0) {
    munmap(const_cast<uint8_t*>(map->data), static_cast<size_t>(map->size));
  }
  *map = {};
}

#endif

} // namespace astral::platform
