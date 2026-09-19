#!/usr/bin/env python3
"""Check one coherent gfx1151 SDK against the source-versioned release pin."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check(home, compiler, pin, environ):
    home = home.resolve(strict=True)
    compiler = compiler.resolve(strict=True)
    if compiler != (home / 'bin/hipcc').resolve(strict=True):
        raise ValueError('HIPCC and ROCM_HOME select different installations')
    for name in ('DS4_ROCM_HOME', 'HIP_PATH', 'ROCM_PATH'):
        if environ.get(name) and Path(environ[name]).resolve() != home:
            raise ValueError(f'{name} selects a different installation from ROCM_HOME')
    for name in ('HIP_CLANG_PATH', 'HIP_LIB_PATH'):
        if environ.get(name):
            Path(environ[name]).resolve().relative_to(home)
    version = subprocess.check_output([str(compiler), '--version'],
        env=environ, text=True, stderr=subprocess.STDOUT)
    match = re.search(r'^HIP version: (\S+)', version, re.M)
    if not match:
        raise ValueError('hipcc did not report a HIP version')
    sdk = (home / '.info/version').read_text().strip()
    mismatches = []
    if sdk != pin['sdk_version']:
        mismatches.append(f'SDK {sdk}, expected {pin["sdk_version"]}')
    if match[1] != pin['hip_version']:
        mismatches.append(f'HIP {match[1]}, expected {pin["hip_version"]}')
    # An explicit old-toolchain comparison still requires one coherent root.
    # It never qualifies as the pinned release; paths must still share a root.
    diagnostic = environ.get('DS4_ALLOW_ROCM_MISMATCH') == '1'
    for name, expected in pin['component_sha256'].items():
        path = home / name
        path.resolve(strict=True).relative_to(home)
        if not diagnostic and sha256(path) != expected:
            mismatches.append(f'component checksum differs: {name}')
    if mismatches and not diagnostic:
        raise ValueError('; '.join(mismatches) +
            '; select the pinned SDK or set DS4_ALLOW_ROCM_MISMATCH=1 for diagnostics')
    return sdk, match[1], diagnostic


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--hipcc', type=Path, required=True)
    args = parser.parse_args()
    pin = json.loads(Path(__file__).with_name('rocm-toolchain.lock.json').read_text())
    try:
        sdk, hip, diagnostic = check(args.home, args.hipcc, pin, dict(os.environ))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'error: ROCm toolchain: {error}', file=sys.stderr)
        return 2
    if diagnostic:
        print('warning: unpinned ROCm diagnostic override; not release validation', file=sys.stderr)
    print(f'ROCm toolchain: SDK={sdk} HIP={hip} root={args.home.resolve()} '
          f'pin={"diagnostic-override" if diagnostic else pin["archive_sha256"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
