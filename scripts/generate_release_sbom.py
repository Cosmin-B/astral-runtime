#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import sys
from pathlib import Path


def load_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except OSError as exc:
        raise SystemExit(f"[release-sbom] failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[release-sbom] invalid JSON in {path}: {exc}") from exc


def package(name, version, supplier, download_location, license_id, external_refs=None):
    data = {
        "SPDXID": f"SPDXRef-Package-{name.replace('.', '-').replace('/', '-')}",
        "name": name,
        "versionInfo": version or "unknown",
        "supplier": supplier,
        "downloadLocation": download_location,
        "filesAnalyzed": False,
        "licenseConcluded": license_id or "NOASSERTION",
        "licenseDeclared": license_id or "NOASSERTION",
        "copyrightText": "NOASSERTION",
    }
    if external_refs:
        data["externalRefs"] = external_refs
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Generate an SPDX 2.3 JSON SBOM from Astral release metadata."
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Path to dependency-manifest.json")
    parser.add_argument("--out", required=True, type=Path, help="Output path for release-sbom.spdx.json")
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    astral = manifest.get("astral", {})
    engine_packages = manifest.get("engine_packages", {})
    submodules = manifest.get("submodules", [])

    version = str(astral.get("version") or "unknown")
    git_commit = str(astral.get("git_commit") or "unknown")
    created = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    packages = [
        package(
            "AstralRT",
            version,
            "Organization: Astral",
            "NOASSERTION",
            "MIT",
            [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:generic/astralrt@{version}?vcs_url=git%2Bhttps://github.com/Cosmin-B/astral.git@{git_commit}",
                }
            ],
        )
    ]

    for entry in submodules:
        name = str(entry.get("name") or entry.get("path") or "unknown")
        commit = str(entry.get("commit") or "unknown")
        description = str(entry.get("description") or commit)
        path = str(entry.get("path") or "")
        packages.append(
            package(
                name,
                description,
                "NOASSERTION",
                f"git+{path}@{commit}" if path else "NOASSERTION",
                str(entry.get("license") or "NOASSERTION"),
                [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": f"pkg:generic/{name}@{commit}",
                    }
                ],
            )
        )

    unity = engine_packages.get("unity", {})
    packages.append(
        package(
            "com.astral.runtime",
            str(unity.get("version") or "unknown"),
            "Organization: Astral",
            "NOASSERTION",
            "MIT",
        )
    )

    unreal = engine_packages.get("unreal", {})
    packages.append(
        package(
            "AstralRT-Unreal",
            str(unreal.get("version") or "unknown"),
            "Organization: Astral",
            "NOASSERTION",
            "MIT",
        )
    )

    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"AstralRT {version} release SBOM",
        "documentNamespace": f"https://github.com/Cosmin-B/astral/releases/{version}/sbom/{git_commit}",
        "creationInfo": {
            "created": created,
            "creators": ["Tool: scripts/generate_release_sbom.py"],
        },
        "packages": packages,
        "relationships": [
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": item["SPDXID"],
            }
            for item in packages
        ],
    }

    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(sbom, f, indent=2, sort_keys=True)
            f.write("\n")
    except OSError as exc:
        raise SystemExit(f"[release-sbom] failed to write {args.out}: {exc}") from exc

    print(f"[release-sbom] SBOM: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
