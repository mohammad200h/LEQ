"""Symlink (or copy) the best dynamics checkpoint for train_LEQ.py.

Reads best_run.yaml and links:
  OfflineRL-Kit/models/dynamics-ensemble/<seed>/<output_model_name>/
to:
  OfflineRL-Kit/models/dynamics-ensemble/<seed>/<task>/

so train_LEQ.py's hardcoded path finds the hyperparameter-search winner.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
LEQ_ROOT = SCRIPT_DIR.parents[2]
WORKSPACE = LEQ_ROOT.parent
OFFLINERL_ROOT = WORKSPACE / 'OfflineRL-Kit'
DEFAULT_BEST_RUN_PATH = SCRIPT_DIR / 'best_run.yaml'
ENSEMBLE_ROOT = OFFLINERL_ROOT / 'models' / 'dynamics-ensemble'


def load_best_run(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f'best_run.yaml not found: {path}')
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'Expected a mapping in {path}')
    for key in ('task', 'seed', 'output_model_name'):
        if key not in cfg:
            raise KeyError(f'Missing required key {key!r} in {path}')
    return cfg


def link_or_copy(src: Path, dst: Path, *, copy: bool, force: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not force:
            raise FileExistsError(
                f'Destination already exists: {dst}\n'
                f'Remove it or pass --force.'
            )
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Install best Walker2D dynamics model for train_LEQ.py'
    )
    parser.add_argument(
        '--best-run',
        type=Path,
        default=DEFAULT_BEST_RUN_PATH,
        help=f'Path to best_run.yaml (default: {DEFAULT_BEST_RUN_PATH})',
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy the checkpoint directory instead of creating a symlink',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Replace an existing destination path',
    )
    args = parser.parse_args()

    cfg = load_best_run(args.best_run)
    task = str(cfg['task'])
    seed = int(cfg['seed'])
    model_name = str(cfg['output_model_name'])

    src = ENSEMBLE_ROOT / str(seed) / model_name
    dst = ENSEMBLE_ROOT / str(seed) / task

    if not src.is_dir():
        raise FileNotFoundError(
            f'Best model checkpoint not found: {src}\n'
            f'Run the hyperparameter search first, or check best_run.yaml.'
        )
    if not (src / 'dynamics.pth').is_file():
        raise FileNotFoundError(f'Missing dynamics.pth under {src}')

    link_or_copy(src, dst, copy=args.copy, force=args.force)
    action = 'Copied' if args.copy else 'Symlinked'
    print(f'{action}:\n  {src}\n-> {dst}')
    print(
        f'\nTrain with the matching seed:\n'
        f"  cd {LEQ_ROOT}\n"
        f"  PYTHONPATH='.' python3 train/train_LEQ.py "
        f"--env_name={task} --expectile 0.5 --seed {seed}"
    )


if __name__ == '__main__':
    main()
