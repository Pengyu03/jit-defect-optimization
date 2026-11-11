"""Fetch the two exact JITMovingTarget CSVs; never replace an existing input.

Only Python's standard library is required. Existing inputs and provenance are
checked against the same pinned values as the original study downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parent
SOURCES = (
    {
        'name': 'openstack',
        'url': 'https://raw.githubusercontent.com/software-rebels/JITMovingTarget/master/data/openstack.csv',
        'bytes': 4448994,
        'sha256': '2b1d2128bee82e905b0233d6646c3fa88eed033e440482e2f89272a3dd436505',
    },
    {
        'name': 'qt',
        'url': 'https://raw.githubusercontent.com/software-rebels/JITMovingTarget/master/data/qt.csv',
        'bytes': 6546502,
        'sha256': '656e6c1b3f06f2bab3329ae002bccdbbfd709b15f8b65cd2f070bbaa6713b48c',
    },
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, source):
    size = path.stat().st_size
    actual = sha256(path)
    if size != source['bytes'] or actual != source['sha256']:
        raise ValueError(
            f'Input identity mismatch for {path}: bytes={size}, SHA-256={actual}; '
            f'expected bytes={source["bytes"]}, SHA-256={source["sha256"]}. '
            'Existing data will not be overwritten; investigate the differing file.')


def verify_manifest(path):
    if not path.exists():
        return False
    actual = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(actual, list) or len(actual) != len(SOURCES):
        raise ValueError(f'Existing source manifest differs from the pinned study: {path}')
    by_name = {record.get('name'): record for record in actual}
    if len(by_name) != len(SOURCES) or any(by_name.get(s['name']) != s for s in SOURCES):
        raise ValueError(f'Existing source manifest differs from the pinned study: {path}; not overwritten')
    return True


def acquire(directory, source, check_only=False):
    output = directory / (source['name'] + '.csv')
    if output.exists():
        verify(output, source)
        print(f'Verified existing input: {output.resolve()}')
        return
    if check_only:
        raise FileNotFoundError(f'Missing input in check-only mode: {output}')
    temporary = output.with_suffix('.csv.part')
    if temporary.exists():
        raise FileExistsError(f'Inspect the pre-existing incomplete download before retrying: {temporary}')
    created_temporary = False
    try:
        request = urllib.request.Request(source['url'], headers={'User-Agent': 'JIT-Optimization-Reproduction/2.0'})
        with temporary.open('xb') as handle:
            created_temporary = True
            with urllib.request.urlopen(request, timeout=60) as response:
                total = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    total += len(chunk)
                    if total > source['bytes']:
                        raise ValueError('Upstream file exceeds its pinned byte count; download rejected')
                    handle.write(chunk)
        verify(temporary, source)
        # Both files are in one directory. A hard link publishes verified bytes
        # atomically and fails if another process has already created output.
        os.link(temporary, output)
        print(f'Downloaded and verified: {output.resolve()}')
    finally:
        if created_temporary and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data' / 'external_raw')
    parser.add_argument('--check-only', action='store_true', help='Verify existing inputs without network access or writes')
    args = parser.parse_args(argv)
    directory = args.output_dir.resolve()
    if not args.check_only:
        directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'sources.json'
    manifest_exists = verify_manifest(manifest)
    if args.check_only and not manifest_exists:
        raise FileNotFoundError(f'Missing source manifest in check-only mode: {manifest}')
    for source in SOURCES:
        acquire(directory, source, args.check_only)
    if not manifest_exists:
        # Exclusive creation also prevents silently replacing a concurrent edit.
        with manifest.open('x', encoding='utf-8') as handle:
            json.dump(list(SOURCES), handle, indent=2)
            handle.write('\n')
    print('Both external inputs and sources.json match the pinned study.')


if __name__ == '__main__':
    main()
