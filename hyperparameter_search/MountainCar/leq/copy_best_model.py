"""Symlink (or copy) the best LEQ policy checkpoint into the default save_dir.

Reads best_run_<reward_mode>.yaml (or an explicit --best-run path) and links:
  <save_dir from best run>/models/<env>/<seed>/<expectile>/
to:
  LEQ/tmp/EP/models/<env>/<seed>/<expectile>/

so a normal train_LEQ / eval path under ./tmp/EP/ finds the search winner.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
LEQ_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_BEST_RUN_STEM = SCRIPT_DIR / 'best_run.yaml'
DEFAULT_INSTALL_ROOT = LEQ_ROOT / 'tmp' / 'EP'
REWARD_MODES = ('twohot', 'gaussian_joint')


def best_run_path_for_mode(base_path: Path, reward_mode: str) -> Path:
    """Map best_run.yaml → best_run_twohot.yaml / best_run_gaussian_joint.yaml."""
    return base_path.with_name(f'{base_path.stem}_{reward_mode}{base_path.suffix}')


def resolve_best_run_path(
    *,
    best_run: Path | None,
    reward_mode: str,
) -> Path:
    if best_run is not None:
        return best_run
    mode_path = best_run_path_for_mode(DEFAULT_BEST_RUN_STEM, reward_mode)
    if mode_path.is_file():
        return mode_path
    # Older searches only wrote best_run.yaml.
    if reward_mode == 'twohot' and DEFAULT_BEST_RUN_STEM.is_file():
        return DEFAULT_BEST_RUN_STEM
    return mode_path


def load_best_run(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f'best_run.yaml not found: {path}')
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'Expected a mapping in {path}')
    for key in ('env_name', 'seed', 'expectile', 'save_dir', 'output_model_name'):
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


def install_best_run(
    cfg: dict[str, Any],
    *,
    install_root: Path,
    copy: bool,
    force: bool,
) -> None:
    env_name = str(cfg['env_name'])
    seed = int(cfg['seed'])
    expectile = cfg['expectile']
    save_dir = Path(cfg['save_dir'])
    if not save_dir.is_absolute():
        save_dir = LEQ_ROOT / save_dir
    reward_mode = str(cfg.get('reward_mode', 'unknown'))

    src = save_dir / 'models' / env_name / str(seed) / str(expectile)
    dst = install_root / 'models' / env_name / str(seed) / str(expectile)
    if not dst.is_absolute():
        dst = LEQ_ROOT / dst

    if not src.is_dir():
        raise FileNotFoundError(
            f'Best policy checkpoint dir not found: {src}\n'
            f'Run the hyperparameter search first, or check best_run.yaml.'
        )

    link_or_copy(src, dst, copy=copy, force=force)
    action = 'Copied' if copy else 'Symlinked'
    print(f'{action} ({reward_mode}):\n  {src}\n-> {dst}')
    load_dir = cfg.get('dynamics_load_dir')
    load_flag = f' --load_dir={load_dir}' if load_dir else ''
    print(
        f'\nWinning run: {cfg["output_model_name"]} '
        f'(final_score={cfg.get("final_score")})\n'
        f'Replay / continue with:\n'
        f"  cd {LEQ_ROOT}\n"
        f"  PYTHONPATH='.' python3 train/train_LEQ.py "
        f"--env_name={env_name} --expectile {expectile} --seed {seed}"
        f"{load_flag}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Install best MountainCar LEQ policy checkpoint under tmp/EP/'
    )
    parser.add_argument(
        '--best-run',
        type=Path,
        default=None,
        help=(
            'Explicit best-run YAML path. Default: '
            'best_run_<reward_mode>.yaml (falls back to best_run.yaml).'
        ),
    )
    parser.add_argument(
        '--reward-mode',
        type=str,
        choices=list(REWARD_MODES),
        default='twohot',
        help='Which dynamics-head best_run to install (default: twohot)',
    )
    parser.add_argument(
        '--all-reward-modes',
        action='store_true',
        help=(
            'Print winners for every reward mode that has a best_run YAML. '
            'Does not install more than one checkpoint (install path is unique); '
            'use --reward-mode to choose which to install.'
        ),
    )
    parser.add_argument(
        '--install-root',
        type=Path,
        default=DEFAULT_INSTALL_ROOT,
        help=f'Default train_LEQ save_dir root (default: {DEFAULT_INSTALL_ROOT})',
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

    if args.all_reward_modes:
        found = False
        for mode in REWARD_MODES:
            path = resolve_best_run_path(best_run=None, reward_mode=mode)
            if not path.is_file():
                print(f'No best_run for {mode}: missing {path}')
                continue
            cfg = load_best_run(path)
            found = True
            print(
                f'{mode}: {cfg["output_model_name"]} '
                f'(seed={cfg["seed"]}, final_score={cfg.get("final_score")}) '
                f'[{path.name}]'
            )
        if not found:
            raise FileNotFoundError(
                'No best_run_<reward_mode>.yaml files found under '
                f'{SCRIPT_DIR}'
            )
        print(
            '\nInstall one with e.g.:\n'
            '  python3 copy_best_model.py --reward-mode twohot --force\n'
            '  python3 copy_best_model.py --reward-mode gaussian_joint --force'
        )
        return

    path = resolve_best_run_path(
        best_run=args.best_run,
        reward_mode=args.reward_mode,
    )
    cfg = load_best_run(path)
    print(f'Using best-run file: {path}')
    install_best_run(
        cfg,
        install_root=Path(args.install_root),
        copy=args.copy,
        force=args.force,
    )


if __name__ == '__main__':
    main()
