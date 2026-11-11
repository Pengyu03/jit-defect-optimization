"""Download the exact ApacheJIT input and verify its pinned SHA-256 digest."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://raw.githubusercontent.com/hosseinkshvrz/apachejit/master/dataset/apachejit_total.csv"
EXPECTED_SHA256 = "5097cbbbab8c4611a709b3cf95ac5c9fe5f62e619bab081911a2a247ddbed4e0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        actual = sha256(output)
        if actual != EXPECTED_SHA256:
            raise ValueError(f"Existing input has unexpected SHA-256 {actual}; it was not overwritten.")
        print(f"Already verified: {output.resolve()}")
        return
    temporary = output.with_suffix(output.suffix + ".part")
    if temporary.exists():
        raise FileExistsError(f"Temporary file already exists; inspect it before retrying: {temporary}")
    try:
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "ApacheJIT-Optimization-Reproduction/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as handle:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(chunk)
        actual = sha256(temporary)
        if actual != EXPECTED_SHA256:
            raise ValueError(f"Downloaded SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual}. "
                             "The upstream file may have changed; do not silently accept a different dataset.")
        temporary.replace(output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    provenance = {
        "url": SOURCE_URL,
        "sha256": EXPECTED_SHA256,
        "bytes": output.stat().st_size,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.with_suffix(".source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))
    print(f"Saved and verified: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "raw" / "apachejit_total.csv")
    download(parser.parse_args().output)
