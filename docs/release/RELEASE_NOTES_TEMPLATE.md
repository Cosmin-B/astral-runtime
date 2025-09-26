# Astral <version> Release Notes

## Compatibility

- Runtime ABI: <major.minor.patch>
- Supported desktop targets: <os/arch list>
- Unity support: <minimum tested version and package dependencies>
- Unreal support: <UE 5.4+ compatibility evidence and UE 5.7 production-target evidence>
- Compiler/toolchain: <compiler and native or cross toolchain versions>

## Artifacts

- Core runtime zip: <file>
- Unity package zip: <file or waiver>
- Unreal plugin zip: <file or waiver>
- Dependency manifest: `dependency-manifest.json`
- ABI layout report: `abi-layout.json`
- Checksums: `checksums.sha256`
- Signature: <checksums signature file or signed-waiver owner>

## Validation Evidence

- Native debug gate: <command and result>
- Native release gate: <command and result>
- CUDA gate: <runner, driver/toolkit, GPU, command, and result>
- MTMD gate: <vision/audio fixtures, command, and result>
- HF matrix: <manifest, logs, and summary>

## Engine Evidence

- Unity Editor ABI tests: <editor version, command, and result>
- Unreal Automation tests: <UE versions, command, and result>
- Unreal container images: <image tags and digests>

## Rollback

- Previous known-good artifact: <version and checksum>
- Previous dependency pins: <llama.cpp, Tracy, engine package versions>
- Rollback command or release-manager procedure: <link or command>

## Known Gaps

- <gap, owner, waiver, and follow-up issue tracker issue>
