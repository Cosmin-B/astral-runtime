#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote


DEFAULT_REPOSITORY_URL = "https://github.com/Cosmin-B/astral"
REPOSITORY_URL_ENV = "ASTRAL_REPOSITORY_URL"


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


def strip_git_suffix(url):
    return url[:-4] if url.endswith(".git") else url


def normalize_repository_url(raw_url):
    url = raw_url.strip()
    if not url:
        return DEFAULT_REPOSITORY_URL
    if url.startswith("git+"):
        url = url[4:]
    ssh_match = re.fullmatch(r"git@([^:]+):(.+)", url)
    if ssh_match:
        host, path = ssh_match.groups()
        return f"https://{host}/{strip_git_suffix(path).rstrip('/')}"
    ssh_url_match = re.fullmatch(r"ssh://git@([^/]+)/(.+)", url)
    if ssh_url_match:
        host, path = ssh_url_match.groups()
        return f"https://{host}/{strip_git_suffix(path).rstrip('/')}"
    return strip_git_suffix(url).rstrip("/")


def repository_url_from_git(repo_root):
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "config", "--get", "remote.origin.url"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def resolve_repository_url(repo_root):
    override = os.environ.get(REPOSITORY_URL_ENV, "")
    if override:
        return normalize_repository_url(override)
    origin = repository_url_from_git(repo_root)
    if origin:
        return normalize_repository_url(origin)
    return DEFAULT_REPOSITORY_URL


def repository_vcs_url(repository_url):
    if repository_url.endswith(".git"):
        return repository_url
    return f"{repository_url}.git"


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
    repository_url = resolve_repository_url(Path(__file__).resolve().parents[1])
    purl_vcs_url = quote(f"git+{repository_vcs_url(repository_url)}@{git_commit}", safe="/:@.")

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
                    "referenceLocator": f"pkg:generic/astralrt@{version}?vcs_url={purl_vcs_url}",
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
        "documentNamespace": f"{repository_url}/releases/{version}/sbom/{git_commit}",
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
