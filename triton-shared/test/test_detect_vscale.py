"""Tests for CPUBackend SVE vscale detection.

Tests two layers of the design:
1. Init guard: _detect_sve_vscale() is only called on aarch64+SVE;
   sve_vscale is None otherwise.
2. Detection: _detect_sve_vscale() raises RuntimeError on failure
   (no silent fallback that hides config errors leading to runtime crash).
"""
import os
import platform
import subprocess
import unittest
from unittest.mock import patch, MagicMock


class FakeBackend:
    """Minimal stand-in for CPUBackend carrying only what detection needs."""

    def __init__(self, cpu_arch, cpu_features):
        self.cpu_arch = cpu_arch
        self.cpu_features = cpu_features
        # Mirror actual __init__ guard logic
        if self.cpu_arch == "aarch64" and "sve" in self.cpu_features:
            self.sve_vscale = self._detect_sve_vscale()
        else:
            self.sve_vscale = None

    def _detect_sve_vscale(self):
        """Matches CPUBackend._detect_sve_vscale — pure detection, no
        arch guard.  Raises RuntimeError on any failure."""
        try:
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmpdir:
                src = os.path.join(tmpdir, "rdvl.c")
                exe = os.path.join(tmpdir, "rdvl")
                Path(src).write_text(
                    '#include <stdio.h>\n#include <stdint.h>\n'
                    'int main(){uint64_t v;asm volatile("rdvl %0, #1":"=r"(v));'
                    'printf("%lu",v/16);return 0;}\n'
                )
                subprocess.check_call(
                    ["gcc", "-march=armv8-a+sve", src, "-o", exe],
                    stderr=subprocess.DEVNULL,
                )
                result = subprocess.check_output([exe]).decode().strip()
                vscale = int(result)
                return max(vscale, 1)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to detect SVE vscale via rdvl: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# 1. Init guard tests — sve_vscale must be None on unsupported platforms
# ---------------------------------------------------------------------------
class TestInitGuard(unittest.TestCase):
    """Verify that _detect_sve_vscale is NOT called on non-SVE platforms."""

    def test_x86_64_gives_none(self):
        """x86_64: sve_vscale must be None, detection never attempted."""
        backend = FakeBackend("x86_64", ["avx2", "sse4.2"])
        self.assertIsNone(backend.sve_vscale)

    def test_aarch64_no_sve_gives_none(self):
        """aarch64 without SVE feature: sve_vscale must be None."""
        backend = FakeBackend("aarch64", ["neon", "fp16"])
        self.assertIsNone(backend.sve_vscale)

    def test_empty_features_gives_none(self):
        """Empty feature list: sve_vscale must be None."""
        backend = FakeBackend("aarch64", [])
        self.assertIsNone(backend.sve_vscale)


# ---------------------------------------------------------------------------
# 2. Detection failure tests — must raise, never silently fall back
# ---------------------------------------------------------------------------
class TestDetectionFailure(unittest.TestCase):
    """When on aarch64+SVE but detection fails, RuntimeError must propagate."""

    @patch("subprocess.check_call", side_effect=FileNotFoundError("gcc"))
    def test_gcc_not_found_raises(self, _mock):
        """Missing gcc must raise RuntimeError, not return a default."""
        with self.assertRaises(RuntimeError) as ctx:
            FakeBackend("aarch64", ["sve"])
        self.assertIn("rdvl", str(ctx.exception))

    @patch("subprocess.check_call")
    @patch("subprocess.check_output", side_effect=OSError("exec failed"))
    def test_rdvl_exec_failure_raises(self, _m1, _m2):
        """rdvl binary execution failure must raise RuntimeError."""
        with self.assertRaises(RuntimeError) as ctx:
            FakeBackend("aarch64", ["sve"])
        self.assertIn("rdvl", str(ctx.exception))


# ---------------------------------------------------------------------------
# 3. Successful detection tests
# ---------------------------------------------------------------------------
class TestDetectionSuccess(unittest.TestCase):
    """Verify correct parsing of rdvl output."""

    @patch("subprocess.check_call")
    @patch("subprocess.check_output", return_value=b"2")
    def test_vscale_2(self, _m1, _m2):
        """256-bit SVE (vscale=2) is parsed correctly."""
        backend = FakeBackend("aarch64", ["sve"])
        self.assertEqual(backend.sve_vscale, 2)

    @patch("subprocess.check_call")
    @patch("subprocess.check_output", return_value=b"1")
    def test_vscale_1(self, _m1, _m2):
        """128-bit SVE (vscale=1) is parsed correctly."""
        backend = FakeBackend("aarch64", ["sve"])
        self.assertEqual(backend.sve_vscale, 1)

    @patch("subprocess.check_call")
    @patch("subprocess.check_output", return_value=b"4")
    def test_vscale_4(self, _m1, _m2):
        """512-bit SVE (vscale=4) is parsed correctly."""
        backend = FakeBackend("aarch64", ["sve"])
        self.assertEqual(backend.sve_vscale, 4)

    @patch("subprocess.check_call")
    @patch("subprocess.check_output", return_value=b"0")
    def test_zero_clamped_to_1(self, _m1, _m2):
        """vscale=0 (impossible but defensive) is clamped to 1."""
        backend = FakeBackend("aarch64", ["sve"])
        self.assertEqual(backend.sve_vscale, 1)


# ---------------------------------------------------------------------------
# 4. Real hardware test (only runs on aarch64+SVE machines)
# ---------------------------------------------------------------------------
class TestRealHardware(unittest.TestCase):

    @unittest.skipUnless(
        platform.machine() == "aarch64",
        "Real detection test requires aarch64",
    )
    def test_real_hardware_positive(self):
        """On real aarch64+SVE the detected vscale must be >= 1."""
        try:
            cpuinfo = open("/proc/cpuinfo").read()
        except OSError:
            self.skipTest("Cannot read /proc/cpuinfo")
        if "sve" not in cpuinfo.lower():
            self.skipTest("SVE not advertised in cpuinfo")

        backend = FakeBackend("aarch64", ["sve"])
        self.assertIsNotNone(backend.sve_vscale)
        self.assertGreaterEqual(backend.sve_vscale, 1)
        # Current ARM implementations use power-of-two vscale
        self.assertEqual(backend.sve_vscale & (backend.sve_vscale - 1), 0,
                         f"vscale={backend.sve_vscale} is not a power of 2")


if __name__ == "__main__":
    unittest.main()
