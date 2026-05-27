import argparse
import atexit
import json
import os
import signal
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = REPO_ROOT / "benchmark"
DEFAULT_OUTPUT_ROOT = BENCHMARK_DIR / "results"
DEFAULT_JOBS = max(1, min(2, os.cpu_count() or 1))

PRIORITY_OPS = [
    "add",
    "add_",
    "cos",
    "cos_",
    "gelu",
    "gelu_",
    "log",
    "log_softmax",
    "mean",
    "mul",
    "mul_",
    "sort",
    "sqrt_",
    "sub",
    "sub_",
    "sum",
    "tan",
    "tan_",
    "tanh",
    "tanh_",
    "vdot",
    "contiguous",
    "mm",
    "dropout",
    "bmm",
    "addmm",
    "flash_attention_forward",
    "flash_attn_varlen_func",
    "flash_mla",
    "gather",
    "group_norm",
    "index_add",
    "index_add_",
    "instance_norm",
    "layer_norm",
    "nll_loss",
    "relu",
    "relu_",
    "rms_norm",
    "scatter",
    "scatter_",
    "sigmoid",
    "sigmoid_",
    "silu",
    "silu_",
    "sin",
    "sin_",
    "topk",
    "topk_softmax",
    "conv1d",
    "conv2d",
    "conv3d",
    "cross_entropy_loss",
    "cumsum",
    "dot",
]

MARK_ALIASES = {
    "flash_atten_varlen_func": "flash_attn_varlen_func",
    "scatter": "scatter_src",
    "scatter_": "scatter_src_",
}

PRINT_LOCK = threading.Lock()
ACTIVE_PROCESSES_LOCK = threading.Lock()
ACTIVE_PROCESSES: dict[int, tuple[str, subprocess.Popen[str]]] = {}
INTERRUPT_EVENT = threading.Event()


def handle_interrupt(signum: int, frame: Any) -> None:
    INTERRUPT_EVENT.set()
    raise KeyboardInterrupt()


@dataclass
class CommandResult:
    op_name: str
    marker: str
    test_file: str
    status: str
    returncode: int
    note: str | None
    stdout_path: str
    record_path: str | None
    summary_path: str
    num_cases: int
    num_success_cases: int
    mean_speedup: float | None
    min_speedup: float | None
    max_speedup: float | None
    min_speedup_dtype: str | None
    max_speedup_dtype: str | None
    min_speedup_shape: Any
    max_speedup_shape: Any
    benchmark_results: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the high-priority FlagGems benchmark suite."
    )
    parser.add_argument(
        "--ops",
        nargs="*",
        default=None,
        help="Optional subset of operator names to run. Defaults to the full priority list.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Override the benchmark warmup count.",
    )
    parser.add_argument(
        "--iter",
        dest="iterations",
        type=int,
        default=None,
        help="Override the benchmark iteration count.",
    )
    parser.add_argument(
        "--mode",
        choices=["kernel", "operator", "wrapper"],
        default="kernel",
        help="Benchmark mode passed to the existing benchmark suite.",
    )
    parser.add_argument(
        "--level",
        choices=["core", "comprehensive"],
        default="comprehensive",
        help="Benchmark level passed to the existing benchmark suite.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help="Number of benchmark subprocesses to run concurrently.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where run artifacts are written.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run directory name. If it exists, it will be overwritten.",
    )
    return parser.parse_args()


def resolve_requested_ops(requested_ops: list[str] | None) -> list[str]:
    ops = requested_ops or PRIORITY_OPS
    resolved = []
    seen = set()
    for op_name in ops:
        canonical_marker = MARK_ALIASES.get(op_name, op_name)
        key = (op_name, canonical_marker)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(op_name)
    return resolved


def discover_marked_tests() -> dict[str, str]:
    discovered: dict[str, str] = {}
    for path in sorted(BENCHMARK_DIR.glob("test_*.py")):
        content = path.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped.startswith("@pytest.mark."):
                continue
            marker = stripped[len("@pytest.mark.") :].split("(", 1)[0].strip()
            if marker and marker not in discovered:
                discovered[marker] = path.name
    return discovered


def build_run_dir(args: argparse.Namespace) -> Path:
    run_name = args.run_name or datetime.now().strftime("priority_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "ops").mkdir(parents=True, exist_ok=True)
    (run_dir / "_bootstrap").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_sitecustomize(run_dir: Path) -> Path:
    sitecustomize_path = run_dir / "_bootstrap" / "sitecustomize.py"
    sitecustomize_path.write_text(
        "import importlib\n"
        "import os\n"
        "if os.environ.get('FLAGGEMS_DISABLE_COMPLEX_BENCH_DTYPE', '1') == '1':\n"
        "    module = importlib.import_module('benchmark.attri_util')\n"
        "    module.COMPLEX_DTYPES = []\n",
        encoding="utf-8",
    )
    return sitecustomize_path.parent


def build_pytest_args(test_file: str, marker: str, args: argparse.Namespace) -> list[str]:
    pytest_args = [
        test_file,
        "-q",
        "-s",
        "-m",
        marker,
        "--record",
        "log",
        "--mode",
        args.mode,
        "--level",
        args.level,
    ]
    if args.warmup is not None:
        pytest_args.extend(["--warmup", str(args.warmup)])
    if args.iterations is not None:
        pytest_args.extend(["--iter", str(args.iterations)])
    return pytest_args


def expected_record_log_path(pytest_args: list[str]) -> Path:
    cmd_args = [
        arg.replace(".py", "").replace("=", "_").replace("/", "_")
        for arg in pytest_args
    ]
    log_file = "result_{}.log".format("_".join(cmd_args)).replace("_-", "-")
    return BENCHMARK_DIR / log_file


def print_line(line: str) -> None:
    with PRINT_LOCK:
        sys.stdout.write(line)
        sys.stdout.flush()


def register_active_process(prefix: str, process: subprocess.Popen[str]) -> None:
    with ACTIVE_PROCESSES_LOCK:
        ACTIVE_PROCESSES[process.pid] = (prefix, process)


def unregister_active_process(process: subprocess.Popen[str]) -> None:
    with ACTIVE_PROCESSES_LOCK:
        ACTIVE_PROCESSES.pop(process.pid, None)


def terminate_active_processes() -> None:
    INTERRUPT_EVENT.set()

    with ACTIVE_PROCESSES_LOCK:
        processes = list(ACTIVE_PROCESSES.values())

    if not processes:
        return

    print_line("\n[suite] interrupt received, terminating running benchmarks...\n")
    for _, process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    for _, process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


atexit.register(terminate_active_processes)


def stream_subprocess(
    command: list[str], cwd: Path, env: dict[str, str], prefix: str
) -> tuple[int, str]:
    if INTERRUPT_EVENT.is_set():
        return -signal.SIGTERM, ""

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    with ACTIVE_PROCESSES_LOCK:
        if INTERRUPT_EVENT.is_set():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            ACTIVE_PROCESSES[process.pid] = (prefix, process)

    output_lines: list[str] = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print_line(f"[{prefix}] {line}")
            output_lines.append(line)

        return process.wait(), "".join(output_lines)
    finally:
        if process.stdout is not None:
            process.stdout.close()
        unregister_active_process(process)


def parse_record_log(path: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("[INFO] "):
            continue
        body = line[len("[INFO] ") :].strip()
        if not body.startswith("{"):
            continue
        data = json.loads(body)
        if isinstance(data, dict) and "op_name" in data and "result" in data:
            payloads.append(data)
    return payloads


def extract_failure_note(output: str, returncode: int) -> str:
    interesting_prefixes = (
        "ModuleNotFoundError:",
        "ImportError:",
        "RuntimeError:",
        "ValueError:",
        "AssertionError:",
        "Failed:",
    )
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(interesting_prefixes):
            return stripped
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if stripped:
            return f"pytest exit code {returncode}: {stripped[:240]}"
    return f"pytest exit code {returncode}"


def summarize_benchmark_results(benchmark_results: list[dict[str, Any]]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    speedups: list[float] = []
    success_count = 0
    min_case = None
    max_case = None

    for result in benchmark_results:
        for metric in result.get("result", []):
            case = {
                "benchmark_op_name": result.get("op_name"),
                "dtype": result.get("dtype"),
                "shape_detail": metric.get("shape_detail"),
                "latency_base": metric.get("latency_base"),
                "latency": metric.get("latency"),
                "speedup": metric.get("speedup"),
                "error_msg": metric.get("error_msg"),
                "status": "failed" if metric.get("error_msg") else "success",
            }
            cases.append(case)
            if case["status"] == "success":
                success_count += 1
            if case["speedup"] is not None:
                speedups.append(case["speedup"])
                if min_case is None or case["speedup"] < min_case["speedup"]:
                    min_case = case
                if max_case is None or case["speedup"] > max_case["speedup"]:
                    max_case = case

    return {
        "cases": cases,
        "num_cases": len(cases),
        "num_success_cases": success_count,
        "mean_speedup": sum(speedups) / len(speedups) if speedups else None,
        "min_speedup": min(speedups) if speedups else None,
        "max_speedup": max(speedups) if speedups else None,
        "min_speedup_dtype": min_case["dtype"] if min_case else None,
        "max_speedup_dtype": max_case["dtype"] if max_case else None,
        "min_speedup_shape": min_case["shape_detail"] if min_case else None,
        "max_speedup_shape": max_case["shape_detail"] if max_case else None,
    }


def determine_status(
    returncode: int, benchmark_results: list[dict[str, Any]], output: str
) -> tuple[str, str | None]:
    if returncode == 0 and benchmark_results:
        return "passed", None
    if returncode != 0 and benchmark_results:
        return "partial", extract_failure_note(output, returncode)
    lowered = output.lower()
    if returncode == 0 and "skipped" in lowered:
        return "skipped", "benchmark skipped by pytest conditions"
    if returncode != 0:
        return "failed", extract_failure_note(output, returncode)
    return "no-data", "no benchmark JSON records were produced"


def relative_to_run_dir(path: Path | None, run_dir: Path) -> str | None:
    if path is None:
        return None
    return str(path.relative_to(run_dir))


def format_speedup(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}x"


def format_shape_detail(shape_detail: Any) -> str:
    if shape_detail is None:
        return "N/A"
    return json.dumps(shape_detail)


def format_extreme_case(prefix: str, op: dict[str, Any]) -> str:
    speedup = op.get(f"{prefix}_speedup")
    dtype = op.get(f"{prefix}_speedup_dtype")
    shape = op.get(f"{prefix}_speedup_shape")
    if speedup is None:
        return "N/A"
    parts = [format_speedup(speedup)]
    if dtype:
        parts.append(f"on {dtype}")
    if shape is not None:
        parts.append(f"@ {format_shape_detail(shape)}")
    return " ".join(parts)


def format_operator_row(op: dict[str, Any]) -> str:
    cases_display = f"{op['num_success_cases']}/{op['num_cases']} passed"
    note = (op["note"] or "").replace("|", "/")
    return (
        "| {op_name} | {marker} | **{status}** | {cases_display} | {mean_speedup} | "
        "{min_display} | {max_display} | {note} | {summary_path} | {stdout_path} | {record_path} |"
    ).format(
        op_name=op["op_name"],
        marker=op["marker"],
        status=op["status"],
        cases_display=cases_display,
        mean_speedup=format_speedup(op.get("mean_speedup")),
        min_display=format_extreme_case("min", op),
        max_display=format_extreme_case("max", op),
        note=note,
        summary_path=op["summary_path"],
        stdout_path=op["stdout_path"],
        record_path=op["record_path"] or "",
    )


def write_initial_summary_markdown(
    summary_md_path: Path,
    *,
    generated_at: str,
    mode: str,
    level: str,
    jobs: int,
    warmup: int | None,
    iterations: int | None,
    unavailable_ops: list[dict[str, str]],
) -> None:
    lines = []
    lines.append("# FlagGems Priority Benchmark Summary")
    lines.append("")
    lines.append(f"- Generated at: {generated_at}")
    lines.append(f"- Mode: {mode}")
    lines.append(f"- Level: {level}")
    lines.append(f"- Jobs: {jobs}")
    lines.append("- Complex dtypes: disabled in runner bootstrap")
    if warmup is not None:
        lines.append(f"- Warmup override: {warmup}")
    if iterations is not None:
        lines.append(f"- Iteration override: {iterations}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("Pending final aggregation.")
    lines.append("")
    if unavailable_ops:
        lines.append("## Unavailable Operators")
        lines.append("")
        lines.append("| Operator | Marker | Reason |")
        lines.append("| --- | --- | --- |")
        for item in unavailable_ops:
            lines.append(
                f"| {item['op_name']} | {item['marker']} | {item['reason']} |"
            )
        lines.append("")
    lines.append("## All Operators")
    lines.append("")
    lines.append(
        "| Operator | Marker | Status | Cases | Mean Speedup | Min / Shape | Max / Shape | Note | Summary | Raw Output | Raw Log |"
    )
    lines.append(
        "| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |"
    )
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_operator_row(summary_md_path: Path, result: CommandResult) -> None:
    with summary_md_path.open("a", encoding="utf-8") as handle:
        handle.write(format_operator_row(asdict(result)) + "\n")


def run_single_benchmark(
    op_name: str,
    marker: str,
    test_file: str,
    args: argparse.Namespace,
    run_dir: Path,
    bootstrap_dir: Path,
) -> CommandResult:
    if INTERRUPT_EVENT.is_set():
        return CommandResult(
            op_name=op_name,
            marker=marker,
            test_file=test_file,
            status="failed",
            returncode=-signal.SIGTERM,
            note="cancelled by interrupt",
            stdout_path="",
            record_path=None,
            summary_path="",
            num_cases=0,
            num_success_cases=0,
            mean_speedup=None,
            min_speedup=None,
            max_speedup=None,
            min_speedup_dtype=None,
            max_speedup_dtype=None,
            min_speedup_shape=None,
            max_speedup_shape=None,
            benchmark_results=[],
        )

    pytest_args = build_pytest_args(test_file, marker, args)
    expected_log = expected_record_log_path(pytest_args)
    expected_log.unlink(missing_ok=True)

    env = os.environ.copy()
    python_path_entries = [
        str(bootstrap_dir),
        str(REPO_ROOT),
        str(REPO_ROOT / "src"),
    ]
    if env.get("PYTHONPATH"):
        python_path_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path_entries)
    env["FLAGGEMS_DISABLE_COMPLEX_BENCH_DTYPE"] = "1"

    command = [sys.executable, "-m", "pytest", *pytest_args]
    print_line(f"\n[suite] starting {op_name}: {' '.join(command)}\n")
    returncode, raw_output = stream_subprocess(command, BENCHMARK_DIR, env, op_name)

    stdout_path = run_dir / "raw" / f"{op_name}.txt"
    stdout_path.write_text(raw_output, encoding="utf-8")

    record_path = None
    benchmark_results: list[dict[str, Any]] = []
    if expected_log.exists():
        record_path = run_dir / "raw" / f"{op_name}.log"
        shutil.move(str(expected_log), record_path)
        benchmark_results = parse_record_log(record_path)

    status, note = determine_status(returncode, benchmark_results, raw_output)
    summary = summarize_benchmark_results(benchmark_results)

    op_summary_path = run_dir / "ops" / f"{op_name}.json"
    op_summary = {
        "op_name": op_name,
        "marker": marker,
        "test_file": test_file,
        "status": status,
        "returncode": returncode,
        "note": note,
        "stdout_path": relative_to_run_dir(stdout_path, run_dir),
        "record_path": relative_to_run_dir(record_path, run_dir),
        **summary,
    }
    op_summary_path.write_text(json.dumps(op_summary, indent=2), encoding="utf-8")

    print_line(
        f"[suite] completed {op_name}: status={status}"
        + (f", note={note}" if note else "")
        + "\n"
    )

    return CommandResult(
        op_name=op_name,
        marker=marker,
        test_file=test_file,
        status=status,
        returncode=returncode,
        note=note,
        stdout_path=relative_to_run_dir(stdout_path, run_dir) or "",
        record_path=relative_to_run_dir(record_path, run_dir),
        summary_path=relative_to_run_dir(op_summary_path, run_dir) or "",
        num_cases=summary["num_cases"],
        num_success_cases=summary["num_success_cases"],
        mean_speedup=summary["mean_speedup"],
        min_speedup=summary["min_speedup"],
        max_speedup=summary["max_speedup"],
        min_speedup_dtype=summary["min_speedup_dtype"],
        max_speedup_dtype=summary["max_speedup_dtype"],
        min_speedup_shape=summary["min_speedup_shape"],
        max_speedup_shape=summary["max_speedup_shape"],
        benchmark_results=summary["cases"],
    )


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = []
    lines.append("# FlagGems Priority Benchmark Summary")
    lines.append("")
    lines.append(f"- Generated at: {summary['generated_at']}")
    lines.append(f"- Mode: {summary['mode']}")
    lines.append(f"- Level: {summary['level']}")
    lines.append(f"- Jobs: {summary['jobs']}")
    lines.append("- Complex dtypes: disabled in runner bootstrap")
    if summary["warmup"] is not None:
        lines.append(f"- Warmup override: {summary['warmup']}")
    if summary["iterations"] is not None:
        lines.append(f"- Iteration override: {summary['iterations']}")
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    lines.append("| Requested | Executed | Passed | Partial | Failed | Skipped | No Data | Unavailable |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        "| {requested} | {executed} | {passed} | {partial} | {failed} | {skipped} | {no_data} | {unavailable} |".format(
            requested=len(summary["requested_ops"]),
            executed=len(summary["ops"]),
            passed=summary["status_counts"].get("passed", 0),
            partial=summary["status_counts"].get("partial", 0),
            failed=summary["status_counts"].get("failed", 0),
            skipped=summary["status_counts"].get("skipped", 0),
            no_data=summary["status_counts"].get("no-data", 0),
            unavailable=len(summary["unavailable_ops"]),
        )
    )
    lines.append("")

    if summary["unavailable_ops"]:
        lines.append("## Unavailable Operators")
        lines.append("")
        lines.append("| Operator | Marker | Reason |")
        lines.append("| --- | --- | --- |")
        for item in summary["unavailable_ops"]:
            lines.append(
                f"| {item['op_name']} | {item['marker']} | {item['reason']} |"
            )
        lines.append("")

    lines.append("## All Operators")
    lines.append("")
    lines.append(
        "| Operator | Marker | Status | Cases | Mean Speedup | Min / Shape | Max / Shape | Note | Summary | Raw Output | Raw Log |"
    )
    lines.append(
        "| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |"
    )
    for op in summary["ops"]:
        lines.append(format_operator_row(op))
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handle_interrupt)
    requested_ops = resolve_requested_ops(args.ops)
    discovered = discover_marked_tests()
    run_dir = build_run_dir(args)
    bootstrap_dir = write_sitecustomize(run_dir)

    unavailable_ops = []
    runnable_ops = []
    for op_name in requested_ops:
        marker = MARK_ALIASES.get(op_name, op_name)
        test_file = discovered.get(marker)
        if test_file is None:
            unavailable_ops.append(
                {
                    "op_name": op_name,
                    "marker": marker,
                    "reason": "no benchmark marker found in benchmark/test_*.py",
                }
            )
            continue
        runnable_ops.append((op_name, marker, test_file))

    max_jobs = max(
        1,
        min(args.jobs, len(runnable_ops) or 1, max(1, (os.cpu_count() or 1) // 2 or 1)),
    )
    generated_at = datetime.now().isoformat(timespec="seconds")
    summary_md_path = run_dir / "summary.md"

    print_line(
        "[suite] complex benchmark dtypes are disabled in this runner without modifying benchmark/attri_util.py\n"
    )
    print_line(f"[suite] results will be written concurrently to {summary_md_path}\n")
    print_line(f"[suite] running {len(runnable_ops)} benchmark(s) with jobs={max_jobs}\n")
    write_initial_summary_markdown(
        summary_md_path,
        generated_at=generated_at,
        mode=args.mode,
        level=args.level,
        jobs=max_jobs,
        warmup=args.warmup,
        iterations=args.iterations,
        unavailable_ops=unavailable_ops,
    )

    completed_results: list[CommandResult] = []
    executor = ThreadPoolExecutor(max_workers=max_jobs)
    try:
        future_to_op = {
            executor.submit(
                run_single_benchmark,
                op_name,
                marker,
                test_file,
                args,
                run_dir,
                bootstrap_dir,
            ): op_name
            for op_name, marker, test_file in runnable_ops
        }
        for future in as_completed(future_to_op):
            result = future.result()
            completed_results.append(result)
            append_operator_row(summary_md_path, result)
    except KeyboardInterrupt:
        terminate_active_processes()
        executor.shutdown(wait=True, cancel_futures=True)
        return 130
    else:
        executor.shutdown(wait=True)
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)

    status_counts: dict[str, int] = {}
    for result in completed_results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1

    summary = {
        "generated_at": generated_at,
        "mode": args.mode,
        "level": args.level,
        "jobs": max_jobs,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "requested_ops": requested_ops,
        "unavailable_ops": unavailable_ops,
        "status_counts": status_counts,
        "ops": [asdict(result) for result in completed_results],
    }

    summary_json_path = run_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md_path.write_text(render_summary_markdown(summary), encoding="utf-8")

    print_line(f"[suite] wrote combined summary to {summary_json_path}\n")
    print_line(f"[suite] wrote combined markdown to {summary_md_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
