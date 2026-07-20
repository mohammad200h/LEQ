"""Grid search over dynamics training knobs by launching run_dynamics.py once per combo."""

from __future__ import annotations

import argparse
import itertools
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
LEQ_ROOT = SCRIPT_DIR.parents[2]
WORKSPACE = LEQ_ROOT.parent
OFFLINERL_ROOT = WORKSPACE / 'OfflineRL-Kit'
DYNAMICS_SCRIPT = OFFLINERL_ROOT / 'run_example' / 'run_dynamics.py'
DEFAULT_CONFIG = SCRIPT_DIR / 'dynamic_modelf.yaml'
DEFAULT_BEST_RUN_PATH = SCRIPT_DIR / 'best_run.yaml'

# Patience large enough that early stopping never fires before max_epochs.
_EARLY_STOP_DISABLED_PATIENCE = 10**9
# Default patience when early stopping is on but YAML omits an explicit value.
_DEFAULT_EARLY_STOP_PATIENCE = 10

_DEFAULT_WEIGHT_DECAY = [2.5e-5, 5.0e-5, 7.5e-5, 7.5e-5, 1.0e-4]
_DEFAULT_HIDDEN_DIMS = [200, 200, 200, 200]
_HOLDOUT_LOSS_RE = re.compile(r'holdout loss:\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)')


def load_search_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f'Config not found: {config_path}')
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'Expected a mapping in {config_path}')
    return cfg


def iter_search_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if value is None:
        return [None]
    return [value]


def resolve_seeds(cfg: dict[str, Any], seed_override: int | None) -> list[int]:
    """CLI --seed wins; else YAML seeds list; else YAML seed; else [1]."""
    if seed_override is not None:
        return [int(seed_override)]
    if 'seeds' in cfg and cfg['seeds'] is not None:
        return [int(s) for s in iter_search_values(cfg['seeds'])]
    return [int(cfg.get('seed', 1))]


def resolve_max_epochs_since_update(cfg: dict[str, Any]) -> int:
    """Return early-stop patience; disable when early_stopping is false/null."""
    early_stopping = cfg.get('early_stopping', True)
    patience = cfg.get('max_epochs_since_update')

    if early_stopping is False or early_stopping is None:
        return _EARLY_STOP_DISABLED_PATIENCE

    if patience is None:
        return _DEFAULT_EARLY_STOP_PATIENCE
    return int(patience)


def resolve_hidden_dims_grid(cfg: dict[str, Any]) -> list[list[int]]:
    """Parse dynamics_hidden_dims as one architecture or a list of architectures."""
    raw = cfg.get('dynamics_hidden_dims', _DEFAULT_HIDDEN_DIMS)
    if raw is None:
        return [list(_DEFAULT_HIDDEN_DIMS)]
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError('dynamics_hidden_dims must be a non-empty list')

    # Single architecture: [200, 200, 200, 200]
    if all(isinstance(x, (int, float)) for x in raw):
        return [[int(x) for x in raw]]

    # Grid of architectures: [[200, 200, 200, 200], [200, 200]]
    arches: list[list[int]] = []
    for item in raw:
        if not isinstance(item, list) or len(item) == 0:
            raise ValueError(
                f'Each dynamics_hidden_dims entry must be a non-empty list, got {item!r}'
            )
        if not all(isinstance(x, (int, float)) for x in item):
            raise ValueError(f'Hidden dims must be numeric, got {item!r}')
        arches.append([int(x) for x in item])
    return arches


def scale_weight_decay(base: list[float], scale: float) -> list[float]:
    return [float(v) * float(scale) for v in base]


def adapt_weight_decay(base: list[float], n_needed: int) -> list[float]:
    """Resample per-layer weight decay to len(hidden_dims) + 1."""
    if n_needed <= 0:
        raise ValueError(f'n_needed must be positive, got {n_needed}')
    if len(base) == n_needed:
        return [float(v) for v in base]
    if len(base) == 1 or n_needed == 1:
        return [float(base[-1])] * n_needed

    # Piecewise-linear interpolate the OfflineRL-Kit schedule onto the new depth.
    src_x = [i / (len(base) - 1) for i in range(len(base))]
    dst_x = [i / (n_needed - 1) for i in range(n_needed)]
    out: list[float] = []
    for x in dst_x:
        for j in range(len(src_x) - 1):
            if src_x[j] <= x <= src_x[j + 1]:
                t = (x - src_x[j]) / (src_x[j + 1] - src_x[j])
                out.append(float(base[j]) * (1 - t) + float(base[j + 1]) * t)
                break
        else:
            out.append(float(base[-1]))
    return out


def format_lr_tag(lr: float) -> str:
    """Compact LR tag for run names, e.g. 0.001 -> 1e-3, 0.0003 -> 3e-4."""
    return f'{lr:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')


def format_hidden_tag(hidden_dims: list[int]) -> str:
    """Compact architecture tag, e.g. [200, 200, 200, 200] -> 200x4."""
    if len(hidden_dims) == 0:
        return 'empty'
    if len(set(hidden_dims)) == 1:
        return f'{hidden_dims[0]}x{len(hidden_dims)}'
    return 'x'.join(str(d) for d in hidden_dims)


def format_float_tag(value: float) -> str:
    return f'{value:g}'.replace('.', 'p')


def build_run_name(
    prefix: str,
    *,
    dynamics_max_epochs: int,
    dynamics_lr: float,
    wd_scale: float,
    n_ensemble: int,
    n_elites: int,
    hidden_dims: list[int],
    holdout_ratio: float,
    dynamics_batch_size: int,
    logvar_loss_coef: float,
    seed: int,
) -> str:
    lr_tag = format_lr_tag(dynamics_lr)
    wd_tag = f'wd{wd_scale:g}x'.replace('.', 'p')
    h_tag = format_hidden_tag(hidden_dims)
    return (
        f'{prefix}_ep{dynamics_max_epochs}_'
        f'lr{lr_tag}_{wd_tag}_'
        f'ens{n_ensemble}e{n_elites}_h{h_tag}_'
        f'hr{format_float_tag(holdout_ratio)}_'
        f'bs{dynamics_batch_size}_'
        f'lv{format_float_tag(logvar_loss_coef)}_'
        f'seed{seed}'
    )


def parse_holdout_loss(text: str) -> float | None:
    """Return the last elite holdout loss printed by EnsembleDynamics.train."""
    matches = _HOLDOUT_LOSS_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def save_best_run(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)
    print(f'Wrote best run parameters to {path}')


def run_dynamics(
    *,
    task: str,
    seed: int,
    dynamics_max_epochs: int,
    max_epochs_since_update: int,
    dynamics_lr: float,
    dynamics_weight_decay: list[float],
    dynamics_hidden_dims: list[int],
    n_ensemble: int,
    n_elites: int,
    holdout_ratio: float,
    dynamics_batch_size: int,
    logvar_loss_coef: float,
    reward_mode: str,
    num_reward_bins: int,
    reward_loss_weight: float,
    dynamics_loss_weight: float,
    eval_num_trajs: int,
    eval_fixed_trajs: bool,
    eval_freq: int,
    run_name: str,
    wandb_project: str,
    track: bool,
    extra_args: list[str],
) -> float | None:
    if not DYNAMICS_SCRIPT.is_file():
        raise FileNotFoundError(
            f'Dynamics script not found: {DYNAMICS_SCRIPT}. '
            'Run LEQ/copy_dynamic_script_to_offline_rl.sh first.'
        )

    cmd = [
        sys.executable,
        str(DYNAMICS_SCRIPT),
        '--task',
        task,
        '--seed',
        str(seed),
        '--dynamics-max-epochs',
        str(dynamics_max_epochs),
        '--max-epochs-since-update',
        str(max_epochs_since_update),
        '--dynamics-lr',
        str(dynamics_lr),
        '--dynamics-weight-decay',
        *[str(v) for v in dynamics_weight_decay],
        '--dynamics-hidden-dims',
        *[str(v) for v in dynamics_hidden_dims],
        '--n-ensemble',
        str(n_ensemble),
        '--n-elites',
        str(n_elites),
        '--holdout-ratio',
        str(holdout_ratio),
        '--dynamics-batch-size',
        str(dynamics_batch_size),
        '--logvar-loss-coef',
        str(logvar_loss_coef),
        '--reward-mode',
        reward_mode,
        '--num-reward-bins',
        str(num_reward_bins),
        '--reward-loss-weight',
        str(reward_loss_weight),
        '--dynamics-loss-weight',
        str(dynamics_loss_weight),
        '--eval-num-trajs',
        str(eval_num_trajs),
        '--eval-freq',
        str(eval_freq),
        '--output-model-name',
        run_name,
        '--wandb-name',
        run_name,
        '--project',
        wandb_project,
    ]
    if eval_fixed_trajs:
        cmd.append('--eval-fixed-trajs')
    else:
        cmd.append('--no-eval-fixed-trajs')
    cmd.extend(extra_args)
    if track:
        cmd.append('--track')

    print(
        f'\n=== seed={seed} max_epochs={dynamics_max_epochs} '
        f'lr={dynamics_lr} wd={dynamics_weight_decay} '
        f'ens={n_ensemble}/{n_elites} hidden={dynamics_hidden_dims} '
        f'holdout={holdout_ratio} bs={dynamics_batch_size} '
        f'logvar={logvar_loss_coef} reward={reward_mode}/bins={num_reward_bins} '
        f'rw={reward_loss_weight} dw={dynamics_loss_weight} ({run_name}) ==='
    )
    print('cwd:', OFFLINERL_ROOT)
    print(' '.join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(OFFLINERL_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    captured: list[str] = []
    for line in proc.stdout:
        print(line, end='')
        captured.append(line)
    returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    return parse_holdout_loss(''.join(captured))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Grid search Walker2D dynamics via run_dynamics.py '
            '(see dynamic_modelf.yaml for all searchable knobs).'
        )
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=DEFAULT_CONFIG,
        help='YAML with search grids and wandb.project',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Run a single seed only (overrides YAML seeds list)',
    )
    parser.add_argument(
        '--project',
        type=str,
        default=None,
        help='WandB project override (default: wandb.project from config)',
    )
    parser.add_argument(
        '--no-track',
        action='store_true',
        help='Disable Weights & Biases logging (enabled by default)',
    )
    parser.add_argument(
        '--best-run-path',
        type=Path,
        default=DEFAULT_BEST_RUN_PATH,
        help=f'Where to write the best-run YAML (default: {DEFAULT_BEST_RUN_PATH})',
    )
    parser.add_argument(
        'overrides',
        nargs='*',
        help='Extra CLI args forwarded to each run_dynamics.py invocation',
    )
    args = parser.parse_args(argv)

    cfg = load_search_config(args.config)
    prefix = cfg['output_model_name_prefix']
    task = cfg.get('task', 'walker2d-medium-replay-v2')
    seeds = resolve_seeds(cfg, args.seed)
    max_epochs_list = [
        int(v) for v in iter_search_values(cfg.get('dynamics_max_epochs', 250))
    ]
    lr_list = [
        float(v) for v in iter_search_values(cfg.get('dynamics_lr', 1e-3))
    ]
    wd_scales = [
        float(v)
        for v in iter_search_values(cfg.get('dynamics_weight_decay_scale', 1.0))
    ]
    wd_base = [
        float(v)
        for v in (
            cfg.get('dynamics_weight_decay_base') or list(_DEFAULT_WEIGHT_DECAY)
        )
    ]
    n_ensemble_list = [
        int(v) for v in iter_search_values(cfg.get('n_ensemble', 7))
    ]
    n_elites_list = [
        int(v) for v in iter_search_values(cfg.get('n_elites', 5))
    ]
    hidden_dims_list = resolve_hidden_dims_grid(cfg)
    holdout_ratio_list = [
        float(v) for v in iter_search_values(cfg.get('holdout_ratio', 0.2))
    ]
    batch_size_list = [
        int(v) for v in iter_search_values(cfg.get('dynamics_batch_size', 256))
    ]
    logvar_list = [
        float(v) for v in iter_search_values(cfg.get('logvar_loss_coef', 0.01))
    ]
    reward_mode = str(cfg.get('reward_mode', 'twohot'))
    num_reward_bins = int(cfg.get('num_reward_bins', 255))
    reward_loss_weight = float(cfg.get('reward_loss_weight', 1.0))
    dynamics_loss_weight = float(cfg.get('dynamics_loss_weight', 1.0))
    max_epochs_since_update = resolve_max_epochs_since_update(cfg)
    early_stopping = cfg.get('early_stopping', True)
    eval_num_trajs = int(cfg.get('eval_num_trajs', 10))
    eval_fixed_trajs = bool(cfg.get('eval_fixed_trajs', True))
    eval_freq = int(cfg.get('eval_freq', 10))
    wandb_cfg = cfg.get('wandb') or {}
    wandb_project = args.project or wandb_cfg.get(
        'project', 'dynamics_walker2d_hyperparameter_search'
    )
    track = not args.no_track

    combos = list(
        itertools.product(
            seeds,
            max_epochs_list,
            lr_list,
            wd_scales,
            n_ensemble_list,
            n_elites_list,
            hidden_dims_list,
            holdout_ratio_list,
            batch_size_list,
            logvar_list,
        )
    )
    valid_combos = [
        c for c in combos if c[5] <= c[4]  # n_elites <= n_ensemble
    ]
    skipped = len(combos) - len(valid_combos)
    print(
        f'Starting dynamics grid search: prefix={prefix}, '
        f'task={task}, seeds={seeds}, '
        f'dynamics_max_epochs={max_epochs_list}, '
        f'dynamics_lr={lr_list}, '
        f'dynamics_weight_decay_scale={wd_scales}, '
        f'n_ensemble={n_ensemble_list}, n_elites={n_elites_list}, '
        f'dynamics_hidden_dims={hidden_dims_list}, '
        f'holdout_ratio={holdout_ratio_list}, '
        f'dynamics_batch_size={batch_size_list}, '
        f'logvar_loss_coef={logvar_list}, '
        f'reward_mode={reward_mode}, num_reward_bins={num_reward_bins}, '
        f'reward_loss_weight={reward_loss_weight}, '
        f'dynamics_loss_weight={dynamics_loss_weight}, '
        f'early_stopping={cfg.get("early_stopping")}, '
        f'max_epochs_since_update={max_epochs_since_update}, '
        f'eval_num_trajs={eval_num_trajs}, '
        f'eval_fixed_trajs={eval_fixed_trajs}, '
        f'eval_freq={eval_freq}, '
        f'n_combos={len(combos)}, n_valid={len(valid_combos)}, '
        f'skipped_invalid_elite={skipped}, '
        f'project={wandb_project}, track={track}'
    )
    if not valid_combos:
        raise RuntimeError(
            'No valid combos (need n_elites <= n_ensemble for every pair).'
        )

    best: dict[str, Any] | None = None
    for (
        seed,
        dynamics_max_epochs,
        dynamics_lr,
        wd_scale,
        n_ensemble,
        n_elites,
        hidden_dims,
        holdout_ratio,
        dynamics_batch_size,
        logvar_loss_coef,
    ) in valid_combos:
        weight_decay = scale_weight_decay(
            adapt_weight_decay(wd_base, len(hidden_dims) + 1),
            wd_scale,
        )
        run_name = build_run_name(
            prefix,
            dynamics_max_epochs=dynamics_max_epochs,
            dynamics_lr=dynamics_lr,
            wd_scale=wd_scale,
            n_ensemble=n_ensemble,
            n_elites=n_elites,
            hidden_dims=hidden_dims,
            holdout_ratio=holdout_ratio,
            dynamics_batch_size=dynamics_batch_size,
            logvar_loss_coef=logvar_loss_coef,
            seed=seed,
        )
        holdout_loss = run_dynamics(
            task=task,
            seed=seed,
            dynamics_max_epochs=dynamics_max_epochs,
            max_epochs_since_update=max_epochs_since_update,
            dynamics_lr=dynamics_lr,
            dynamics_weight_decay=weight_decay,
            dynamics_hidden_dims=hidden_dims,
            n_ensemble=n_ensemble,
            n_elites=n_elites,
            holdout_ratio=holdout_ratio,
            dynamics_batch_size=dynamics_batch_size,
            logvar_loss_coef=logvar_loss_coef,
            reward_mode=reward_mode,
            num_reward_bins=num_reward_bins,
            reward_loss_weight=reward_loss_weight,
            dynamics_loss_weight=dynamics_loss_weight,
            eval_num_trajs=eval_num_trajs,
            eval_fixed_trajs=eval_fixed_trajs,
            eval_freq=eval_freq,
            run_name=run_name,
            wandb_project=wandb_project,
            track=track,
            extra_args=args.overrides,
        )
        if holdout_loss is None:
            print(
                f'Warning: could not parse holdout loss for {run_name}; '
                'skipping for best-run selection.'
            )
            continue

        print(f'Parsed holdout loss for {run_name}: {holdout_loss}')
        candidate = {
            'task': task,
            'seed': seed,
            'output_model_name': run_name,
            'dynamics_max_epochs': dynamics_max_epochs,
            'dynamics_lr': dynamics_lr,
            'dynamics_weight_decay_scale': wd_scale,
            'dynamics_weight_decay': weight_decay,
            'n_ensemble': n_ensemble,
            'n_elites': n_elites,
            'dynamics_hidden_dims': hidden_dims,
            'holdout_ratio': holdout_ratio,
            'dynamics_batch_size': dynamics_batch_size,
            'logvar_loss_coef': logvar_loss_coef,
            'reward_mode': reward_mode,
            'num_reward_bins': num_reward_bins,
            'reward_loss_weight': reward_loss_weight,
            'dynamics_loss_weight': dynamics_loss_weight,
            'early_stopping': early_stopping,
            'max_epochs_since_update': max_epochs_since_update,
            'eval_num_trajs': eval_num_trajs,
            'eval_fixed_trajs': eval_fixed_trajs,
            'eval_freq': eval_freq,
            'holdout_loss': holdout_loss,
            'wandb': {
                'project': wandb_project,
                'name': run_name,
            },
        }
        if best is None or holdout_loss < float(best['holdout_loss']):
            best = candidate
            print(
                f'New best run: {run_name} '
                f'(holdout_loss={holdout_loss})'
            )

    if best is None:
        raise RuntimeError(
            'No successful runs with a parseable holdout loss; '
            f'not writing {args.best_run_path}'
        )
    save_best_run(args.best_run_path, best)


if __name__ == '__main__':
    main()
