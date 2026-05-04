import argparse
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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

    errors: list[str] = []

    def run_one(book_dir: Path, i: int) -> str | None:
        # Example format requested: (1/100) process 書名1 ...
        # Returns the failed book name (folder name) or None on success.
        progress_state = "none" if args.force else classify_book_progress(book_dir, required_finish_code)
        if progress_state == "done":
            tqdm.write(f"({i}/{total}) skip done {book_dir.name} ...")
            return None

        # For "partial"/"none" we just re-run the book. To keep this wrapper
        # compatibility with upstream (no core code changes), we only skip
        # fully finished books based on stored progress.
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
        try:
            subprocess.run(cmd, check=True, timeout=float(args.timeout) or None, env=env)
            return None
        except subprocess.TimeoutExpired:
            tqdm.write(f"({i}/{total}) timeout {book_dir.name} after {args.timeout:g}s")
            return book_dir.name
        except subprocess.CalledProcessError:
            tqdm.write(f"({i}/{total}) failed {book_dir.name}")
            return book_dir.name
        except Exception:
            tqdm.write(f"({i}/{total}) failed {book_dir.name}")
            return book_dir.name

    if concurrency == 1:
        # Sequential mode preserves processing order (and keeps output readable).
        for i, book_dir in enumerate(tqdm(book_dirs, total=total, unit="book"), start=1):
            failed = run_one(book_dir, i)
            if failed is not None:
                errors.append(failed)
    else:
        # Parallel mode: each book is processed by a separate `launch.py` process.
        # NOTE: output from multiple processes may interleave in the console.
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = []
            for i, book_dir in enumerate(book_dirs, start=1):
                futures.append(ex.submit(run_one, book_dir, i))

            for fut in tqdm(as_completed(futures), total=total, unit="book"):
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
