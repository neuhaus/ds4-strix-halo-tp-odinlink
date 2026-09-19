#!/usr/bin/env python3
"""Exercise the actual SDK pin checker without installing a compiler."""
import hashlib
from pathlib import Path
import runpy
import tempfile
import unittest

CHECK = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/check-rocm-toolchain.py'))['check']


class ToolchainPinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / 'sdk'
        (self.home / 'bin').mkdir(parents=True)
        (self.home / '.info').mkdir()
        (self.home / 'lib').mkdir()
        (self.home / '.info/version').write_text('10.0.0\n')
        self.compiler = self.home / 'bin/hipcc'
        self.compiler.write_text("#!/bin/sh\nprintf 'HIP version: 7.15.26333-0000000\\n'\n")
        self.compiler.chmod(0o755)
        self.library = self.home / 'lib/libamdhip64.so'
        self.library.write_bytes(b'pinned-runtime')
        self.pin = dict(sdk_version='10.0.0', hip_version='7.15.26333-0000000',
            component_sha256={name: hashlib.sha256((self.home / name).read_bytes()).hexdigest()
                              for name in ('bin/hipcc', 'lib/libamdhip64.so')})
        self.env = dict(PATH='/usr/bin:/bin')

    def check(self):
        return CHECK(self.home, self.compiler, self.pin, self.env)

    def test_sdk10_hip715_is_valid(self):
        self.assertEqual(self.check(), ('10.0.0', '7.15.26333-0000000', False))

    def test_reject_wrong_sdk(self):
        (self.home / '.info/version').write_text('7.14.0\n')
        with self.assertRaisesRegex(ValueError, 'SDK 7.14.0'):
            self.check()

    def test_reject_modified_runtime(self):
        self.library.write_bytes(b'another-release')
        with self.assertRaisesRegex(ValueError, 'component checksum differs'):
            self.check()

    def test_reject_compiler_from_other_install(self):
        external = Path(self.tmp.name) / 'hipcc'
        external.write_bytes(self.compiler.read_bytes())
        external.chmod(0o755)
        self.compiler = external
        with self.assertRaisesRegex(ValueError, 'different installations'):
            self.check()

    def test_reject_environment_from_other_install(self):
        for name in ('DS4_ROCM_HOME', 'HIP_PATH', 'ROCM_PATH'):
            with self.subTest(name=name):
                self.env[name] = '/other-rocm'
                with self.assertRaisesRegex(ValueError, 'different installation'):
                    self.check()
                del self.env[name]

    def test_reject_external_library_symlink_even_with_override(self):
        external = Path(self.tmp.name) / 'runtime'
        external.write_bytes(self.library.read_bytes())
        self.library.unlink()
        self.library.symlink_to(external)
        self.env['DS4_ALLOW_ROCM_MISMATCH'] = '1'
        with self.assertRaises(ValueError):
            self.check()

    def test_comparison_override_is_explicitly_unpinned(self):
        (self.home / '.info/version').write_text('7.14.0\n')
        self.env['DS4_ALLOW_ROCM_MISMATCH'] = '1'
        self.assertTrue(self.check()[2])


if __name__ == '__main__':
    unittest.main()
