#!/usr/bin/env python3
"""
Refactored: run all reconstruction scripts with 5 epochs each.
- Preserves original behaviour and defaults.
- Improves readability, structure, and maintainability.
"""

from __future__ import annotations

import logging
import os
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple


# -----------------------------------------------------------------------------
# Configuration containers
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    script_dir: Path
    config_dir: Path
    output_base: Path
    num_epochs: int = 3
    timeout_s: int = 1800  # 30 minutes


@dataclass(frozen=True)
class TestCase:
    script_rel: str  # e.g. "run_hkem_1bpos.py"
    config_filename: str  # e.g. "config_hkem_1bpos_pet.yaml"
    name: str  # e.g. "hkem_1bpos_pet_hybrid"
    overrides: Mapping[str, object]  # extra overrides, e.g. {"method": "ista"}

    def script_path(self, settings: Settings) -> Path:
        return settings.script_dir / self.script_rel

    def config_path(self, settings: Settings) -> Path:
        return settings.config_dir / self.config_filename


@dataclass
class TestResult:
    name: str
    script_rel: str
    config_filename: str
    ok: bool


# -----------------------------------------------------------------------------
# Test matrix (unchanged semantics; just structured)
# -----------------------------------------------------------------------------
RAW_TESTS: Mapping[str, Sequence[Tuple[str, str, Mapping[str, object]]]] = {
    # DTNV reconstructions - only use proper 1bpos/2bpos configs
    "run_dtnv_1bpos.py": [
        ("config_1bpos_anthro.yaml", "dtnv_1bpos_anthro", {}),
    ],
    "run_dtnv_2bpos.py": [
        ("config_2bpos.yaml", "dtnv_2bpos", {}),
    ],
    # HKEM reconstructions (hybrid and non-hybrid)
    "run_hkem_1bpos.py": [
        ("config_hkem_1bpos_pet.yaml", "hkem_1bpos_pet_hybrid", {"method": "ista"}),
        (
            "config_hkem_1bpos_pet.yaml",
            "hkem_1bpos_pet_kosmaposl",
            {"method": "kosmaposl"},
        ),
        ("config_hkem_1bpos_spect.yaml", "hkem_1bpos_spect_hybrid", {"method": "ista"}),
        ("config_kem_1bpos_pet.yaml", "kem_1bpos_pet_nonhybrid", {"method": "ista"}),
    ],
    "run_hkem_2bpos.py": [
        ("config_hkem_2bpos.yaml", "hkem_2bpos_hybrid", {}),
        ("config_kem_2bpos.yaml", "kem_2bpos_nonhybrid", {}),
    ],
    # MyKEM - should only be KEM (non-hybrid)
    "run_hkem_1bpos_my_kem.py": [
        ("config_kem_1bpos_pet.yaml", "mykem_1bpos_pet", {"method": "ista"}),
    ],
}


def build_tests(
    raw: Mapping[str, Sequence[Tuple[str, str, Mapping[str, object]]]],
) -> List[TestCase]:
    tests: List[TestCase] = []
    for script_rel, items in raw.items():
        tests.extend(
            TestCase(script_rel, config_filename, name, dict(overrides))
            for config_filename, name, overrides in items
        )
    return tests


# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def test_output_dirs(base: Path, test_name: str) -> Tuple[Path, Path]:
    out_dir = base / test_name
    work_dir = out_dir / "working"
    ensure_dir(work_dir)
    return out_dir, work_dir


def build_command(
    py_exe: str, settings: Settings, tc: TestCase, out_dir: Path, work_dir: Path
) -> List[str]:
    cmd: List[str] = [
        py_exe,
        str(tc.script_path(settings)),
        "--config",
        str(tc.config_path(settings)),
        "--override",
        f"num_epochs={settings.num_epochs}",
        "--override",
        f"output_path={out_dir}",
        "--override",
        f"working_path={work_dir}",
    ]
    for k, v in tc.overrides.items():
        cmd.extend(["--override", f"{k}={v}"])
    return cmd


def write_test_log(
    out_dir: Path,
    test_name: str,
    cmd: Sequence[str],
    duration_s: float,
    proc: subprocess.CompletedProcess[str],
) -> None:
    log_file = out_dir / "test_log.txt"
    with open(log_file, "w") as f:
        f.write(f"Test: {test_name}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Duration: {duration_s:.2f} seconds\n")
        f.write(f"Return code: {proc.returncode}\n\n")
        f.write("STDOUT:\n")
        f.write(proc.stdout)
        f.write("\nSTDERR:\n")
        f.write(proc.stderr)


def write_timeout_log(
    out_dir: Path,
    test_name: str,
    cmd: Sequence[str],
    duration_s: float,
    e: subprocess.TimeoutExpired,
) -> None:
    log_file = out_dir / "test_log.txt"
    with open(log_file, "w") as f:
        f.write(f"Test: {test_name}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Duration: {duration_s:.2f} seconds (TIMEOUT)\n")
        f.write(f"Status: TIMEOUT after {int(duration_s)} seconds\n\n")
        f.write("STDOUT (partial):\n")
        f.write(e.stdout if isinstance(e.stdout, str) and e.stdout else "No stdout captured\n")
        f.write("\nSTDERR (partial):\n")
        f.write(e.stderr if isinstance(e.stderr, str) and e.stderr else "No stderr captured\n")


# -----------------------------------------------------------------------------
# Core execution
# -----------------------------------------------------------------------------


def run_test(settings: Settings, tc: TestCase) -> TestResult:
    out_dir, work_dir = test_output_dirs(settings.output_base, tc.name)
    cmd = build_command(sys.executable, settings, tc, out_dir, work_dir)

    logging.info("==== Running: %s", tc.name)
    logging.info("Script: %s", tc.script_path(settings))
    logging.info("Config: %s", tc.config_path(settings))
    logging.info("Command: %s", " ".join(cmd))

    start = time.time()
    stdout_lines = []
    stderr_lines = []

    try:
        # Use Popen for real-time output capture with unbuffered output
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered output

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            env=env,
        )

        # Real-time output capture with timeout

        # Set stdout and stderr to non-blocking
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stderr.fileno(), False)

        timeout_end = start + settings.timeout_s

        while process.poll() is None and time.time() < timeout_end:
            # Check for available output
            ready, _, _ = select.select([process.stdout, process.stderr], [], [], 1.0)

            for stream in ready:
                if stream == process.stdout:
                    if line := stream.readline():
                        stdout_lines.append(line)
                        logging.debug("STDOUT: %s", line.rstrip())
                elif stream == process.stderr:
                    if line := stream.readline():
                        stderr_lines.append(line)
                        logging.debug("STDERR: %s", line.rstrip())

        # Handle timeout
        if time.time() >= timeout_end and process.poll() is None:
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()
            # Collect any remaining output
            remaining_stdout, remaining_stderr = process.communicate()
            if remaining_stdout:
                stdout_lines.append(remaining_stdout)
            if remaining_stderr:
                stderr_lines.append(remaining_stderr)

            duration = time.time() - start
            # Create a fake TimeoutExpired with collected output
            fake_timeout = subprocess.TimeoutExpired(cmd, settings.timeout_s)
            fake_timeout.stdout = "".join(stdout_lines)
            fake_timeout.stderr = "".join(stderr_lines)
            write_timeout_log(out_dir, tc.name, cmd, duration, fake_timeout)
            logging.error("⏰ TIMEOUT: %s exceeded %d seconds", tc.name, settings.timeout_s)
            ok = False
        else:
            # Process completed normally, collect any remaining output
            remaining_stdout, remaining_stderr = process.communicate()
            if remaining_stdout:
                stdout_lines.append(remaining_stdout)
            if remaining_stderr:
                stderr_lines.append(remaining_stderr)

            duration = time.time() - start

            # Create a fake CompletedProcess for compatibility
            fake_proc = subprocess.CompletedProcess(
                cmd, process.returncode, "".join(stdout_lines), "".join(stderr_lines)
            )
            write_test_log(out_dir, tc.name, cmd, duration, fake_proc)

            if process.returncode == 0:
                logging.info("✅ SUCCESS: %s (%.2fs)", tc.name, duration)
                ok = True
            else:
                snippet = ("".join(stderr_lines))[:200]
                logging.error("❌ FAILED: %s (rc=%s) | %s", tc.name, process.returncode, snippet)
                ok = False

    except Exception as e:  # noqa: BLE001
        duration = time.time() - start
        logging.exception("💥 ERROR: %s crashed after %.2fs | %s", tc.name, duration, e)
        ok = False

    return TestResult(tc.name, tc.script_rel, tc.config_filename, ok)


def write_summary(summary_path: Path, settings: Settings, results: Sequence[TestResult]) -> None:
    total = len(results)
    successful = sum(bool(r.ok) for r in results)
    with open(summary_path, "w") as f:
        f.write("Reconstruction Test Summary\n")
        f.write("===========================\n\n")
        f.write(f"Test epochs: {settings.num_epochs}\n")
        f.write(f"Total tests: {total}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {total - successful}\n")
        f.write(f"Success rate: {successful / total * 100:.1f}%\n\n")
        f.write("Detailed Results:\n")
        for r in results:
            status = "PASS" if r.ok else "FAIL"
            f.write(f"{status}: {r.name} ({r.script_rel} with {r.config_filename})\n")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> None:
    setup_logging()

    settings = Settings(
        script_dir=Path("/home/sam/working/synergistic_recon/scripts"),
        config_dir=Path("/home/sam/working/synergistic_recon/configs"),
        output_base=Path("/home/sam/working/synergistic_recon/test_results"),
        num_epochs=5,
        timeout_s=3600,  # Increase to 1 hour
    )

    ensure_dir(settings.output_base)

    tests = build_tests(RAW_TESTS)
    total = len(tests)

    logging.info("🧪 Starting comprehensive reconstruction testing")
    logging.info("Test epochs: %d", settings.num_epochs)
    logging.info("Output directory: %s", settings.output_base)
    logging.info("Planned tests: %d", total)

    results: List[TestResult] = []
    for i, tc in enumerate(tests, start=1):
        logging.info("Progress: %d/%d — %s", i, total, tc.name)
        results.append(run_test(settings, tc))

    # Summary
    successful = sum(bool(r.ok) for r in results)
    logging.info(
        "🏁 TEST SUMMARY: %d/%d passed (%.1f%%)",
        successful,
        total,
        successful / total * 100,
    )
    for r in results:
        logging.info("  %s %s (%s)", "✅ PASS" if r.ok else "❌ FAIL", r.name, r.script_rel)

    summary_path = settings.output_base / "test_summary.txt"
    write_summary(summary_path, settings, results)
    logging.info("Summary saved to: %s", summary_path)
    logging.info("Individual test results in: %s", settings.output_base)

    # Exit with non-zero code if any tests failed
    if successful < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
