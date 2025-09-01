# External Dependencies

Astral expects some third-party dependencies to live under `astral/external/` (preferred) so builds and CI are reproducible.

## llama.cpp

Preferred strategy: add llama.cpp as a git submodule at `astral/external/llama.cpp`.

Example:

```bash
cd astral
git submodule add https://github.com/ggml-org/llama.cpp.git external/llama.cpp
git submodule update --init --recursive
```

`astral/src/backend/CMakeLists.txt` prefers `external/llama.cpp` automatically, and falls back to a sibling checkout at `../llama.cpp` for local dev.
