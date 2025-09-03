# bench/run_project_sharded.py
import argparse, os, shutil, subprocess, sys, time
from pathlib import Path
from math import ceil

IMG_EXTS = {".png", ".jpg", ".jpeg"}

def list_images(root: Path):
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])

def make_shards(files, n_shards, tmp_root: Path):
    tmp_root.mkdir(parents=True, exist_ok=True)
    shards = [[] for _ in range(n_shards)]
    for i, f in enumerate(files):
        shards[i % n_shards].append(f)
    shard_dirs = []
    for i, shard in enumerate(shards):
        sd = tmp_root / f"shard_{i:02d}"
        sd.mkdir(exist_ok=True, parents=True)
        # symlink files zodat upstream script gewoon --dir kan gebruiken
        for f in shard:
            dst = sd / f.name
            try:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(f.resolve())
            except OSError:
                # op sommige systemen geen symlinks → kopiëren als fallback
                shutil.copy2(f, dst)
        shard_dirs.append(sd)
    return shard_dirs

def run_one_shard(repo_root: Path, shard_dir: Path, env):
    cmd = [sys.executable, "chess_diagram_to_fen.py", "--dir", str(shard_dir)]
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, cwd=str(repo_root), env=env)

def main():
    ap = argparse.ArgumentParser(description="Parallel sharded runner voor chess_diagram_to_fen.py")
    ap.add_argument("--input", required=True, help="Map met png/jpg/jpeg diagrammen")
    ap.add_argument("--workers", type=int, default=2, help="# parallelle processen (2–3 is meestal goed op M1)")
    ap.add_argument("--out", default="bench/out_project_sharded.txt", help="Gecombineerde log")
    ap.add_argument("--scratch", default="bench/_shards", help="Tijdelijke shard-map (symlinks/kopieën)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    inp = Path(args.input)
    if not inp.is_dir():
        raise SystemExit("Geef een map door aan --input")

    files = list_images(inp)
    if not files:
        raise SystemExit("Geen afbeeldingen gevonden.")

    # maak shards
    scratch = Path(args.scratch)
    if scratch.exists():
        shutil.rmtree(scratch)
    shard_dirs = make_shards(files, max(1, args.workers), scratch)

    # ENV: non-interactive & sitecustomize
    env = os.environ.copy()
    env["NOGUI"] = "1"
    env["PYTHONPATH"] = str(repo_root / "speedkit") + (
        os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    t0 = time.perf_counter()
    outs = []
    errs = 0

    procs = []
    for sd in shard_dirs:
        procs.append(
            subprocess.Popen(
                [sys.executable, "chess_diagram_to_fen.py", "--dir", str(sd)],
                cwd=str(repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )

    # verzamel output
    for p in procs:
        out, _ = p.communicate()
        if p.returncode != 0:
            errs += 1
        outs.append(out)

    dt = time.perf_counter() - t0

    # log
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n\n".join(outs), encoding="utf-8")

    ms_per_item = (1000.0 * dt / len(files))
    print(f"Files: {len(files)} | workers: {args.workers} | total: {dt:.2f}s | ~{ms_per_item:.1f} ms/item | shard fails: {errs}")
    print(f"→ log: {args.out}")

if __name__ == "__main__":
    main()
