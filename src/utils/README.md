# Runtime utilities

The utility layer contains three bounded building blocks used by the core:

- `utf8.hpp` and `utf8.cpp`: byte-span validation and code-point counting,
  with runtime-selected scalar, x86, and ARM paths.
- `string_builder.hpp`: an append-only builder with explicit spill or truncate
  behavior.
- `logging.hpp` and `logging.cpp`: callback-based logging with runtime-owned
  formatting storage.

`trace.hpp` supplies the optional Tracy boundary used by maintained hot-path
zones. `test_utils.cpp` exercises UTF-8, string-builder, and logging contracts.

These are internal C++ APIs. Public callers use `AstralSpanU8`,
`AstralMutSpanU8`, and the logging fields on `AstralInit` from
[`include/astral_rt.h`](../../include/astral_rt.h).

Run the maintained test target through the repository preset:

```bash
ctest --preset release-with-tests -R '^test_utils$' --output-on-failure
```
