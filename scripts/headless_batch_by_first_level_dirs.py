import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Thread

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
ANSI = {
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "gray": "\033[90m",
    "reset": "\033[0m",
}


def configure_console_encoding() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_gui_default_config():
    """Load GUI default config (the same file BallonsTranslator uses when you don't pass --config_path)."""
    from utils import config as program_config
    from utils import shared

    program_config.load_config(shared.CONFIG_PATH)
    return program_config.pcfg


def get_sakura_api_base(pcfg) -> str:
    """
    Extract Sakura api base url from loaded config.
    Expected shape (as produced by GUI): pcfg.module.translator_params['Sakura']['api baseurl']['value']
    """
    module = pcfg.module
    translator_params = module.get_params("translator") or {}
    sakura_params = translator_params.get("Sakura") or {}

    raw = None
    v = sakura_params.get("api baseurl", None)
    if isinstance(v, dict) and "value" in v:
        raw = v["value"]
    else:
        raw = v

    if not raw or not isinstance(raw, str):
        raise SystemExit("Could not find Sakura 'api baseurl' in config. Please verify GUI translator params.")

    url = raw.strip()
    if url.endswith("/"):
        url = url[:-1]
    # Normalize to ".../v1"
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def iter_configured_devices(pcfg):
    for module_type in ("textdetector", "ocr", "inpainter"):
        module_name = getattr(pcfg.module, module_type, None)
        if not module_name:
            continue
        params = (pcfg.module.get_params(module_type) or {}).get(module_name) or {}
        for key, value in params.items():
            raw = value.get("value") if isinstance(value, dict) and "value" in value else value
            if key == "device" and isinstance(raw, str) and raw:
                yield module_type, module_name, raw


def pretest_torch_devices(pcfg) -> None:
    """Fail fast when config requests CUDA but PyTorch cannot use CUDA."""
    configured_devices = list(iter_configured_devices(pcfg))
    cuda_users = [
        f"{module_type}:{module_name}={device}"
        for module_type, module_name, device in configured_devices
        if device.lower().startswith("cuda")
    ]
    if not cuda_users:
        return

    try:
        import torch
    except Exception as e:
        raise SystemExit(f"CUDA device configured but PyTorch cannot be imported. Modules: {', '.join(cuda_users)}. Error: {e}") from e

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA device configured but PyTorch CUDA is unavailable.\n"
            f"Modules: {', '.join(cuda_users)}\n"
            f"torch: {getattr(torch, '__version__', 'unknown')}\n"
            f"torch.version.cuda: {getattr(torch.version, 'cuda', None)}\n"
            "Install a CUDA-enabled PyTorch build for this environment, or change these module devices to CPU."
        )


def pretest_sakura_endpoint(api_base: str, timeout_s: float) -> None:
    """Fail fast if Sakura server is not reachable."""
    import openai

    major = int(openai.__version__.split(".")[0])
    api_key = os.environ.get("SAKURA_API_KEY", "not-required")
    messages = [{"role": "user", "content": "ping"}]

    try:
        if major >= 1:
            client = openai.Client(api_key=api_key, base_url=api_base)
            client.chat.completions.create(
                model="sukinishiro",
                messages=messages,
                temperature=0.1,
                top_p=0.3,
                max_tokens=1,
                frequency_penalty=0.05,
                seed=-1,
                extra_query={"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0},
            )
        else:
            openai.api_base = api_base
            openai.api_key = api_key
            openai.ChatCompletion.create(
                model="sukinishiro",
                messages=messages,
                temperature=0.1,
                top_p=0.3,
                max_tokens=1,
                frequency_penalty=0.05,
                seed=-1,
                extra_query={"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0},
            )
    except Exception as e:
        raise SystemExit(f"Sakura pretest failed: cannot reach {api_base}. Error: {e}") from e

def load_required_finish_code_optional() -> int:
    """
    Required finish_code bitmask used to decide "book already done".
    In real usage it comes from GUI config; in dev/test env where config may be missing,
    fallback to RunStatus.FIN_ALL so we don't crash.
    """
    try:
        pcfg = load_gui_default_config()
        return int(getattr(pcfg.module, "finish_code", 15))
    except Exception:
        try:
            from utils.config import RunStatus
            return int(RunStatus.FIN_ALL)
        except Exception:
            return 15


def book_proj_json_path(book_dir: Path) -> Path:
    """
    ProjImgTrans uses:
      proj_name() = 'imgtrans_' + basename(directory)
      proj_path = directory / (proj_name + '.json')
    """
    proj_name = "imgtrans_" + book_dir.name
    return book_dir / (proj_name + ".json")


def classify_book_progress(book_dir: Path, required_finish_code: int) -> str:
    """
    Returns one of:
      - 'done' (all pages finished)
      - 'partial' (some pages finished, some not)
      - 'none' (no prior progress detected / no json)
    """
    import json

    proj_path = book_proj_json_path(book_dir)
    if not proj_path.exists():
        return "none"

    try:
        data = json.loads(proj_path.read_text(encoding="utf8"))
    except Exception:
        return "none"

    image_info = data.get("image_info", {}) or {}
    pages = data.get("pages", {}) or {}
    if not pages:
        return "none"

    finished_flags = []
    for page_name in pages.keys():
        fin_code = 0
        try:
            fin_code = int(image_info.get(page_name, {}).get("finish_code", 0))
        except Exception:
            fin_code = 0
        finished_flags.append((fin_code & required_finish_code) == required_finish_code)

    if not any(finished_flags):
        return "none"
    if all(finished_flags):
        return "done"
    return "partial"


def count_book_images(book_dir: Path) -> int:
    return sum(1 for path in book_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def read_book_progress_counts(book_dir: Path, required_finish_code: int) -> tuple[int, int]:
    import json

    proj_path = book_proj_json_path(book_dir)
    if not proj_path.exists():
        return 0, count_book_images(book_dir)

    try:
        data = json.loads(proj_path.read_text(encoding="utf8"))
    except Exception:
        return 0, count_book_images(book_dir)

    image_info = data.get("image_info", {}) or {}
    pages = data.get("pages", {}) or {}
    total = len(pages) or count_book_images(book_dir)
    finished = 0
    for page_name in pages.keys():
        try:
            fin_code = int(image_info.get(page_name, {}).get("finish_code", 0))
        except Exception:
            fin_code = 0
        if (fin_code & required_finish_code) == required_finish_code:
            finished += 1
    return finished, total


def safe_log_name(name: str) -> str:
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name).strip(" .")
    return safe_name[:120] or "book"


def tail_text_file(path: Path, max_lines: int = 40) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    return lines[-max_lines:]


def supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def color_status(status: str, color_enabled: bool) -> str:
    if not color_enabled:
        return status
    color = {
        "done": "green",
        "failed": "red",
        "timeout": "red",
        "running": "yellow",
        "queued": "gray",
        "skipped": "gray",
    }.get(status)
    if color is None:
        return status
    return f"{ANSI[color]}{status}{ANSI['reset']}"


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def shorten_text(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."


def render_progress_table(states: list[dict], concurrency: int, started_at: float, color_enabled: bool) -> str:
    done_count = sum(1 for state in states if state["status"] in {"done", "skipped"})
    failed_count = sum(1 for state in states if state["status"] in {"failed", "timeout"})
    running_count = sum(1 for state in states if state["status"] == "running")
    summary_parts = [f"{done_count}/{len(states)} done"]
    if running_count:
        summary_parts.append(f"{running_count} running")
    if failed_count:
        summary_parts.append(f"{failed_count} failed")
    summary_parts.append(f"concurrency={concurrency}")
    summary_parts.append(f"elapsed {format_elapsed(time.monotonic() - started_at)}")

    lines = [f"Books: {' | '.join(summary_parts)}", ""]
    lines.append(f"{'#':<4}{'Status':<12}{'Pages':>10}  {'Percent':>7}  {'Time':>8}  Book")
    for state in states:
        total = max(int(state.get("total") or 0), 0)
        finished = max(int(state.get("finished") or 0), 0)
        percent = int(finished / total * 100) if total else 0
        pages = f"{finished}/{total}" if total else "0/?"
        elapsed = format_elapsed((state.get("ended_at") or time.monotonic()) - state["started_at"]) if state.get("started_at") else "00:00"
        raw_status = state["status"]
        status = color_status(raw_status, color_enabled)
        status_cell = f"{status}{' ' * max(0, 12 - len(raw_status))}"
        lines.append(
            f"{state['index']:<4}{status_cell}{pages:>10}  {percent:>6}%  {elapsed:>8}  {shorten_text(state['book'], 56)}"
        )
    return "\n".join(lines)


def table_progress_worker(states: list[dict], lock: Lock, stop_event: dict, concurrency: int, started_at: float) -> None:
    color_enabled = supports_color()
    use_repaint = sys.stdout.isatty()
    last_render_line_count = 0
    first_render = True
    while not stop_event["stop"]:
        with lock:
            snapshot = [state.copy() for state in states]
        output = render_progress_table(snapshot, concurrency, started_at, color_enabled)
        if not use_repaint:
            print(output, "\n", flush=True)
        elif first_render:
            print(output, end="", flush=True)
            first_render = False
        else:
            print(f"\033[{last_render_line_count}F\033[J{output}", end="", flush=True)
        if use_repaint:
            last_render_line_count = output.count("\n") + 1
        time.sleep(1)

    with lock:
        snapshot = [state.copy() for state in states]
    output = render_progress_table(snapshot, concurrency, started_at, color_enabled)
    if not use_repaint:
        print(output, flush=True)
    elif first_render:
        print(output, flush=True)
    else:
        print(f"\033[{last_render_line_count}F\033[J{output}", flush=True)


def run_child_with_table_progress(
    cmd: list[str],
    env: dict[str, str],
    timeout: float | None,
    log_file,
    book_dir: Path,
    required_finish_code: int,
    state: dict,
    lock: Lock,
) -> None:
    started_at = time.monotonic()
    finished, total = read_book_progress_counts(book_dir, required_finish_code)
    with lock:
        state.update({"status": "running", "started_at": started_at, "finished": finished, "total": total})

    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    try:
        while proc.poll() is None:
            if timeout is not None and time.monotonic() - started_at > timeout:
                proc.kill()
                proc.wait()
                raise subprocess.TimeoutExpired(cmd, timeout)

            finished, total = read_book_progress_counts(book_dir, required_finish_code)
            with lock:
                state.update({"finished": finished, "total": total})
            time.sleep(0.5)

        finished, total = read_book_progress_counts(book_dir, required_finish_code)
        with lock:
            state.update({"finished": finished, "total": total})

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    finally:
        if proc.poll() is None:
            proc.kill()


def parse_args() -> argparse.Namespace:
    # Wrapper parameters:
    # - --root: the folder that contains all "book" subfolders
    # - --concurrency: how many headless instances to run at the same time
    p = argparse.ArgumentParser(
        description="Run launch.py in headless mode for each first-level subdirectory under a root directory."
    )
    p.add_argument(
        "--root",
        required=True,
        help='Root directory path (e.g. "C:\\Users\\USER\\Downloads\\comic"). First-level subfolders are treated as books.',
    )
    p.add_argument(
        "--skip-pretest",
        action="store_true",
        default=False,
        help="Skip Sakura endpoint pretest (not recommended).",
    )
    p.add_argument(
        "--pretest-timeout",
        type=float,
        default=10.0,
        help="Timeout seconds for Sakura endpoint pretest.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="How many headless instances to run in parallel. 1 = sequential (default).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="Per-book timeout in seconds. 0 = no timeout (default).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Process every book even if existing project metadata says it is done.",
    )
    p.add_argument(
        "--simple-progress",
        action="store_true",
        default=False,
        help="Only show book-level progress. Child launch.py output is written to logs/headless_batch.",
    )
    p.add_argument(
        "--sort",
        action="store_true",
        default=True,
        help="Sort book folders by name before processing.",
    )
    p.add_argument(
        "--no-sort",
        action="store_false",
        dest="sort",
        help="Do not sort book folders.",
    )
    return p.parse_args()


def main() -> int:
    configure_console_encoding()
    args = parse_args()

    root = Path(args.root).expanduser()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"--root is not a directory: {root}")

    # This script treats each first-level folder as a separate "exec_dir"
    # (which BallonsTranslator headless expects).
    book_dirs = [p for p in root.iterdir() if p.is_dir()]
    if args.sort:
        book_dirs.sort(key=lambda x: x.name.casefold())

    concurrency = max(1, int(args.concurrency))
    total = len(book_dirs)

    if not args.skip_pretest:
        pcfg = load_gui_default_config()
        pretest_torch_devices(pcfg)
        api_base = get_sakura_api_base(pcfg)
        print(f"Pretesting Sakura endpoint: {api_base} ...")
        pretest_sakura_endpoint(api_base=api_base, timeout_s=float(args.pretest_timeout))
    else:
        pretest_torch_devices(load_gui_default_config())

    required_finish_code = load_required_finish_code_optional()

    repo_root = Path(__file__).resolve().parents[1]
    launch_py = repo_root / "launch.py"
    if not launch_py.exists():
        raise SystemExit(f"launch.py not found at expected path: {launch_py}")

    log_dir = repo_root / "logs" / "headless_batch"
    errors: list[str] = []

    def run_one(book_dir: Path, i: int, books_bar=None, state=None, state_lock=None) -> str | None:
        # Example format requested: (1/100) process 書名1 ...
        # Returns the failed book name (folder name) or None on success.
        progress_state = "none" if args.force else classify_book_progress(book_dir, required_finish_code)
        if progress_state == "done":
            if state is not None and state_lock is not None:
                finished, page_total = read_book_progress_counts(book_dir, required_finish_code)
                with state_lock:
                    state.update({
                        "status": "skipped",
                        "finished": finished,
                        "total": page_total,
                        "started_at": time.monotonic(),
                        "ended_at": time.monotonic(),
                    })
            elif books_bar is not None:
                books_bar.set_description_str(f"Books {i}/{total} skip")
                books_bar.set_postfix_str(book_dir.name[:40], refresh=True)
            else:
                tqdm.write(f"({i}/{total}) skip done {book_dir.name} ...")
            return None

        # For "partial"/"none" we just re-run the book. To keep this wrapper
        # compatibility with upstream (no core code changes), we only skip
        # fully finished books based on stored progress.
        if state is not None and state_lock is not None:
            finished, page_total = read_book_progress_counts(book_dir, required_finish_code)
            with state_lock:
                state.update({"status": "running", "finished": finished, "total": page_total, "started_at": time.monotonic()})
        elif books_bar is not None:
            books_bar.set_description_str(f"Books {i}/{total} processing")
            books_bar.set_postfix_str(book_dir.name[:40], refresh=True)
        else:
            tqdm.write(f"({i}/{total}) process {book_dir.name} ...")
        cmd = [
            sys.executable,
            str(launch_py),
            "--headless",
            "--exec_dirs",
            str(book_dir),
        ]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        log_path = None
        subprocess_kwargs = {
            "check": True,
            "timeout": float(args.timeout) or None,
            "env": env,
        }
        log_file = None
        if args.simple_progress:
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"{i:04d}_{timestamp}_{safe_log_name(book_dir.name)}.log"
            log_file = log_path.open("w", encoding="utf-8", errors="replace")
            subprocess_kwargs["stdout"] = log_file
            subprocess_kwargs["stderr"] = subprocess.STDOUT
        try:
            if args.simple_progress and log_file is not None:
                if state is None or state_lock is None:
                    raise RuntimeError("simple progress requires state tracking")
                run_child_with_table_progress(
                    cmd=cmd,
                    env=env,
                    timeout=float(args.timeout) or None,
                    log_file=log_file,
                    book_dir=book_dir,
                    required_finish_code=required_finish_code,
                    state=state,
                    lock=state_lock,
                )
                with state_lock:
                    state.update({"status": "done", "ended_at": time.monotonic()})
            else:
                subprocess.run(cmd, **subprocess_kwargs)
            return None
        except subprocess.TimeoutExpired:
            if state is not None and state_lock is not None:
                with state_lock:
                    state.update({"status": "timeout", "ended_at": time.monotonic()})
            elif books_bar is not None:
                books_bar.set_description_str(f"Books {i}/{total} timeout")
                books_bar.set_postfix_str(book_dir.name[:40], refresh=True)
            else:
                tqdm.write(f"({i}/{total}) timeout {book_dir.name} after {args.timeout:g}s")
            if log_path is not None:
                tqdm.write(f"  log: {log_path}")
            return book_dir.name
        except subprocess.CalledProcessError:
            if state is not None and state_lock is not None:
                with state_lock:
                    state.update({"status": "failed", "ended_at": time.monotonic()})
            elif books_bar is not None:
                books_bar.set_description_str(f"Books {i}/{total} failed")
                books_bar.set_postfix_str(book_dir.name[:40], refresh=True)
            else:
                tqdm.write(f"({i}/{total}) failed {book_dir.name}")
            if log_path is not None:
                if log_file is not None:
                    log_file.flush()
                tqdm.write(f"  log: {log_path}")
                for line in tail_text_file(log_path):
                    tqdm.write(f"  {line}")
            return book_dir.name
        except Exception:
            if state is not None and state_lock is not None:
                with state_lock:
                    state.update({"status": "failed", "ended_at": time.monotonic()})
            elif books_bar is not None:
                books_bar.set_description_str(f"Books {i}/{total} failed")
                books_bar.set_postfix_str(book_dir.name[:40], refresh=True)
            else:
                tqdm.write(f"({i}/{total}) failed {book_dir.name}")
            if log_path is not None:
                tqdm.write(f"  log: {log_path}")
            return book_dir.name
        finally:
            if log_file is not None:
                log_file.close()

    if args.simple_progress:
        states = []
        for i, book_dir in enumerate(book_dirs, start=1):
            _, page_total = read_book_progress_counts(book_dir, required_finish_code)
            states.append({
                "index": i,
                "book": book_dir.name,
                "status": "queued",
                "finished": 0,
                "total": page_total,
                "started_at": None,
                "ended_at": None,
            })

        state_lock = Lock()
        stop_event = {"stop": False}
        table_started_at = time.monotonic()
        renderer = Thread(
            target=table_progress_worker,
            args=(states, state_lock, stop_event, concurrency, table_started_at),
            daemon=True,
        )
        renderer.start()
        try:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = {}
                for i, book_dir in enumerate(book_dirs, start=1):
                    futures[ex.submit(run_one, book_dir, i, None, states[i - 1], state_lock)] = (i, book_dir)

                for fut in as_completed(futures):
                    _, _book_dir = futures[fut]
                    failed = fut.result()
                    if failed is not None:
                        errors.append(failed)
        finally:
            stop_event["stop"] = True
            renderer.join()
    elif concurrency == 1:
        # Sequential mode preserves processing order (and keeps output readable).
        with tqdm(total=total, unit="book", desc="Books", position=0, dynamic_ncols=True) as books_bar:
            for i, book_dir in enumerate(book_dirs, start=1):
                failed = run_one(book_dir, i, books_bar if args.simple_progress else None)
                books_bar.update(1)
                if args.simple_progress:
                    status = "failed" if failed is not None else "done"
                    books_bar.set_description_str(f"Books {i}/{total} {status}")
                    books_bar.set_postfix_str(book_dir.name[:40], refresh=True)
                if failed is not None:
                    errors.append(failed)
    else:
        # Parallel mode: each book is processed by a separate `launch.py` process.
        # NOTE: output from multiple processes may interleave in the console.
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = []
            for i, book_dir in enumerate(book_dirs, start=1):
                futures.append(ex.submit(run_one, book_dir, i))

            for fut in tqdm(as_completed(futures), total=total, unit="book", desc="Books"):
                failed = fut.result()
                if failed is not None:
                    errors.append(failed)

    if errors:
        print("")
        print(f"Failed books: {len(errors)}")
        for name in errors:
            print(f"- {name}")
        return 1

    print("")
    print("All books completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
