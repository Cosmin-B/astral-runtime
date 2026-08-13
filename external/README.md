# External Dependencies

Astral pins third-party source dependencies as submodules under
`astral-runtime/external/`.

## llama.cpp

Initialize the pinned llama.cpp and Tracy revisions from a public clone:

```bash
cd astral-runtime
git submodule update --init --recursive
```

The build prefers `external/llama.cpp` and can use a sibling `../llama.cpp`
checkout for local development. Do not update either submodule as part of an
unrelated Astral change.
