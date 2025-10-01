# Dependency Manifest

Astral release artifacts must be traceable to exact source and dependency pins.

## Current Pins

Committed pins live in `docs/release/dependency-pins.tsv` and are checked by:

```bash
./scripts/validate_dependency_pins.sh
```

| Dependency | Pin source | Current value |
|---|---|---|
| Astral runtime | `CMakeLists.txt`, `include/astral_rt.h` | `0.1.0` |
| llama.cpp | Git submodule `external/llama.cpp` | `b9025`, `eff06702b2a52e1020ea009ebd86cb9f5acabab5` |
| Tracy | Git submodule `external/tracy` | `a602127eddb60825ac91e726986c12955e9a0082` |
| Unity package | `plugins/unity/package.json` | `com.astral.runtime` `0.1.0`, Unity `2021.3` |
| Unreal package | `plugins/unreal/AstralRT/AstralRT.uplugin` | UE 5.4+ compatibility floor; UE 5.7 is the production target |

## Generated Manifest

Generate the release dependency manifest and artifact checksums after packaging:

```bash
./scripts/generate_abi_layout_report.sh --out dist/abi-layout.json
./scripts/generate_release_metadata.sh dist
./scripts/validate_release_artifacts.sh --dist dist
```

The release packaging flow writes:

- `dist/abi-layout.json`
- `dist/dependency-manifest.json`
- `dist/checksums.sha256`, covering packaged zips, `abi-layout.json`, and `dependency-manifest.json`
- `dist/checksums.sha256.asc` when signed with GPG
- `dist/checksums.sha256.minisig` when signed with minisign

The generated manifest records the Astral version, source commit, dirty state,
submodule commits, and engine package versions. `abi-layout.json` records public
C ABI struct sizes and alignments for the release host. `checksums.sha256`
covers files already present in the output directory, including the generated
dependency manifest.

## Signing

Sign the checksum file for a release candidate:

```bash
ASTRAL_RELEASE_SIGN_KEY=release@example.com ./scripts/sign_release_artifacts.sh --out-dir dist
```

Verify the signed artifact set:

```bash
gpg --verify dist/checksums.sha256.asc dist/checksums.sha256
(cd dist && sha256sum -c checksums.sha256)
```

Release managers own the private signing key. CI may sign only from a protected
release environment; presubmit and local smoke jobs should use
`scripts/sign_release_artifacts.sh --dry-run`.

The protected CI signing lane is `.github/workflows/release-sign.yml`. It is
manual-only, runs in the `release` environment, validates release notes and
dependency pins, downloads artifact zips plus metadata from a completed artifact
run, signs every `checksums.sha256`, verifies the signatures, and uploads the
signed manifests. Configure the environment with:

- `ASTRAL_RELEASE_GPG_PRIVATE_KEY`
- `ASTRAL_RELEASE_GPG_PASSPHRASE`
- `ASTRAL_RELEASE_GPG_KEY_ID`

Release storage policy:

- Store `checksums.sha256` and its detached signature next to the release zips.
- Publish the public verification key or GPG fingerprint in the release notes.
- Keep private keys in the protected release secret store only; never in the repo,
  build cache, artifact zips, or local developer defaults.
- Treat a missing signature as a release blocker unless the release notes carry an
  explicit waiver owned by the release manager.
