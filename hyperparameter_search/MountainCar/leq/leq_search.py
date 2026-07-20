"""Grid search over LEQ policy knobs by launching train_LEQ.py once per combo.

Each combo is trained once per dynamics reward head (twohot / gaussian_joint),
loading the frozen winner from ../dynamic_model/best_run_<reward_mode>.yaml
via train_LEQ.py --load_dir (no need to reinstall the task symlink).
"""

from __future__ import annotations

import argparse
import itertools
import os
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
ENSEMBLE_ROOT = OFFLINERL_ROOT / 'models' / 'dynamics-ensemble'
DYNAMICS_BEST_RUN_STEM = (
    SCRIPT_DIR.parent / 'dynamic_model' / 'best_run.yaml'
)
TRAIN_SCRIPT = LEQ_ROOT / 'train' / 'train_LEQ.py'
DEFAULT_CONFIG = SCRIPT_DIR / 'leq.yaml'
DEFAULT_BEST_RUN_PATH = SCRIPT_DIR / 'best_run.yaml'
DEFAULT_SAVE_ROOT = LEQ_ROOT / 'tmp' / 'EP' / 'leq_search'
REWARD_MODES = ('twohot', 'gaussian_joint')

_FINAL_SCORE_RE = re.compile(
    r'final score:\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'
)
_STEP_RETURN_RE = re.compile(
    r'^Step\s+(\d+)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'
)


def best_run_path_for_mode(base_path: Path, reward_mode: str) -> Path:
    """Map best_run.yaml → best_run_twohot.yaml / best_run_gaussian_joint.yaml."""
    return base_path.with_name(f'{base_path.stem}_{reward_mode}{base_path.suffix}')


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


def format_lr_tag(lr: float) -> str:
    """Compact LR tag for run names, e.g. 0.001 -> 1e-3, 3e-5 -> 3e-5."""
    return f'{lr:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')


def format_float_tag(value: float) -> str:
    return f'{value:g}'.replace('.', 'p')


def format_reward_mode_tag(reward_mode: str) -> str:
    if reward_mode == 'gaussian_joint':
        return 'rmgauss'
    if reward_mode == 'twohot':
        return 'rmtwohot'
    return f'rm{reward_mode}'


def resolve_dynamics_best_run_path(
    *,
    reward_mode: str,
    dynamics_best_run_stem: Path,
) -> Path:
    mode_path = best_run_path_for_mode(dynamics_best_run_stem, reward_mode)
    if mode_path.is_file():
        return mode_path
    # Older searches only wrote best_run.yaml for the twohot winner.
    if reward_mode == 'twohot' and dynamics_best_run_stem.is_file():
        return dynamics_best_run_stem
    return mode_path


def load_dynamics_checkpoint(
    *,
    reward_mode: str,
    dynamics_best_run_stem: Path,
) -> dict[str, Any]:
    """Resolve frozen dynamics path + metadata for one reward head."""
    best_path = resolve_dynamics_best_run_path(
        reward_mode=reward_mode,
        dynamics_best_run_stem=dynamics_best_run_stem,
    )
    if not best_path.is_file():
        raise FileNotFoundError(
            f'Missing dynamics best-run YAML for reward_mode={reward_mode}: '
            f'{best_path}\n'
            'Finish ../dynamic_model/dynamic_model_search.py first, or pass '
            '--dynamics-best-run-stem pointing at the best_run.yaml stem.'
        )
    with best_path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'Expected a mapping in {best_path}')
    for key in ('task', 'seed', 'output_model_name'):
        if key not in cfg:
            raise KeyError(f'Missing required key {key!r} in {best_path}')

    dyn_seed = int(cfg['seed'])
    model_name = str(cfg['output_model_name'])
    load_dir = ENSEMBLE_ROOT / str(dyn_seed) / model_name
    if not load_dir.is_dir():
        raise FileNotFoundError(
            f'Dynamics checkpoint not found for {reward_mode}: {load_dir}\n'
            f'(from {best_path})'
        )
    if not (load_dir / 'dynamics.pth').is_file():
        raise FileNotFoundError(f'Missing dynamics.pth under {load_dir}')

    return {
        'reward_mode': str(cfg.get('reward_mode', reward_mode)),
        'best_run_path': str(best_path),
        'dynamics_seed': dyn_seed,
        'dynamics_model_name': model_name,
        'load_dir': str(load_dir.resolve()),
        'holdout_loss': cfg.get('holdout_loss'),
    }


def build_run_name(
    prefix: str,
    *,
    expectile: float,
    model_batch_ratio: float,
    rollout_length: int,
    horizon_length: int,
    lamb: float,
    actor_lr: float,
    discount: float,
    batch_size: int,
    pretrain: bool,
    reward_mode: str,
    seed: int,
) -> str:
    return (
        f'{prefix}_'
        f'exp{format_float_tag(expectile)}_'
        f'mbr{format_float_tag(model_batch_ratio)}_'
        f'rl{rollout_length}_'
        f'hz{horizon_length}_'
        f'lam{format_float_tag(lamb)}_'
        f'alr{format_lr_tag(actor_lr)}_'
        f'g{format_float_tag(discount)}_'
        f'bs{batch_size}_'
        f'pt{int(pretrain)}_'
        f'{format_reward_mode_tag(reward_mode)}_'
        f'seed{seed}'
    )


def parse_final_score(text: str) -> float | None:
    """Return final score printed by train_LEQ, else last eval Step return."""
    matches = _FINAL_SCORE_RE.findall(text)
    if matches:
        return float(matches[-1])

    last: float | None = None
    for line in text.splitlines():
        m = _STEP_RETURN_RE.match(line.strip())
        if m is not None:
            last = float(m.group(2))
    return last


def save_best_run(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)
    print(f'Wrote best run parameters to {path}')


def run_leq(
    *,
    env_name: str,
    seed: int,
    expectile: float,
    model_batch_ratio: float,
    rollout_length: int,
    horizon_length: int,
    lamb: float,
    actor_lr: float,
    value_lr: float,
    critic_lr: float,
    discount: float,
    batch_size: int,
    max_steps: int,
    num_layers: int,
    layer_size: int,
    rollout_batch_size: int,
    rollout_freq: int,
    rollout_retain: int,
    num_repeat: int,
    actor_update: str,
    critic_update: str,
    pretrain: bool,
    video_interval: int,
    eval_interval: int,
    save_interval: int,
    load_dir: str,
    run_name: str,
    save_dir: Path,
    wandb_project: str,
    track: bool,
    wandb_key: str,
    extra_args: list[str],
) -> float | None:
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(f'Train script not found: {TRAIN_SCRIPT}')

    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        f'--env_name={env_name}',
        f'--seed={seed}',
        f'--expectile={expectile}',
        f'--model_batch_ratio={model_batch_ratio}',
        f'--rollout_length={rollout_length}',
        f'--horizon_length={horizon_length}',
        f'--lamb={lamb}',
        f'--discount={discount}',
        f'--batch_size={batch_size}',
        f'--max_steps={max_steps}',
        f'--num_layers={num_layers}',
        f'--layer_size={layer_size}',
        f'--rollout_batch_size={rollout_batch_size}',
        f'--rollout_freq={rollout_freq}',
        f'--rollout_retain={rollout_retain}',
        f'--num_repeat={num_repeat}',
        f'--actor_update={actor_update}',
        f'--critic_update={critic_update}',
        f'--video_interval={video_interval}',
        f'--eval_interval={eval_interval}',
        f'--save_interval={save_interval}',
        f'--save_dir={save_dir}',
        f'--load_dir={load_dir}',
        f'--config.actor_lr={actor_lr}',
        f'--config.value_lr={value_lr}',
        f'--config.critic_lr={critic_lr}',
        f'--wandb_project={wandb_project}',
        f'--wandb_name={run_name}',
    ]
    if pretrain:
        cmd.append('--pretrain')
    if track:
        if wandb_key:
            cmd.append(f'--wandb_key={wandb_key}')
    else:
        cmd.append('--debug')
    cmd.extend(extra_args)

    print(
        f'\n=== seed={seed} expectile={expectile} '
        f'mbr={model_batch_ratio} rollout={rollout_length} '
        f'horizon={horizon_length} lamb={lamb} actor_lr={actor_lr} '
        f'discount={discount} bs={batch_size} pretrain={pretrain} '
        f'load_dir={load_dir} ({run_name}) ==='
    )
    print('cwd:', LEQ_ROOT)
    print(' '.join(cmd))
    env = {**os.environ, 'PYTHONPATH': str(LEQ_ROOT)}
    proc = subprocess.Popen(
        cmd,
        cwd=str(LEQ_ROOT),
        env=env,
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
    return parse_final_score(''.join(captured))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Grid search MountainCar LEQ via train_LEQ.py '
            '(see leq.yaml for all searchable knobs). '
            'Uses frozen dynamics winners from ../dynamic_model/ '
            'for each reward_mode.'
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
        help=(
            'Base path for best-run YAMLs; one file is written per reward_mode '
            f'(e.g. best_run_twohot.yaml). Default: {DEFAULT_BEST_RUN_PATH}'
        ),
    )
    parser.add_argument(
        '--dynamics-best-run-stem',
        type=Path,
        default=DYNAMICS_BEST_RUN_STEM,
        help=(
            'Stem of dynamics best_run YAML '
            f'(default: {DYNAMICS_BEST_RUN_STEM}); '
            'resolved as best_run_<reward_mode>.yaml'
        ),
    )
    parser.add_argument(
        '--save-root',
        type=Path,
        default=None,
        help=(
            'Root directory for per-run save_dir folders '
            f'(default: {DEFAULT_SAVE_ROOT})'
        ),
    )
    parser.add_argument(
        '--wandb-key',
        type=str,
        default='',
        help='WandB API key forwarded to train_LEQ.py when tracking',
    )
    parser.add_argument(
        'overrides',
        nargs='*',
        help='Extra CLI args forwarded to each train_LEQ.py invocation',
    )
    args = parser.parse_args(argv)

    cfg = load_search_config(args.config)
    prefix = cfg['output_model_name_prefix']
    env_name = cfg.get('env_name', cfg.get('task', 'mountaincar-human-v0'))
    seeds = resolve_seeds(cfg, args.seed)

    expectile_list = [
        float(v) for v in iter_search_values(cfg.get('expectile', 0.5))
    ]
    model_batch_ratio_list = [
        float(v) for v in iter_search_values(cfg.get('model_batch_ratio', 0.25))
    ]
    rollout_length_list = [
        int(v) for v in iter_search_values(cfg.get('rollout_length', 5))
    ]
    horizon_length_list = [
        int(v) for v in iter_search_values(cfg.get('horizon_length', 10))
    ]
    lamb_list = [float(v) for v in iter_search_values(cfg.get('lamb', 0.95))]
    actor_lr_list = [
        float(v) for v in iter_search_values(cfg.get('actor_lr', 3e-5))
    ]
    value_lr = float(cfg.get('value_lr', 1e-4))
    critic_lr = float(cfg.get('critic_lr', value_lr))
    discount_list = [
        float(v) for v in iter_search_values(cfg.get('discount', 0.997))
    ]
    batch_size_list = [
        int(v) for v in iter_search_values(cfg.get('batch_size', 256))
    ]
    pretrain_list = [
        bool(v) for v in iter_search_values(cfg.get('pretrain', False))
    ]
    reward_mode_list = [
        str(v) for v in iter_search_values(cfg.get('reward_mode', list(REWARD_MODES)))
    ]

    max_steps = int(cfg.get('max_steps', int(1e6)))
    num_layers = int(cfg.get('num_layers', 3))
    layer_size = int(cfg.get('layer_size', 256))
    rollout_batch_size = int(cfg.get('rollout_batch_size', 50000))
    rollout_freq = int(cfg.get('rollout_freq', 1000))
    rollout_retain = int(cfg.get('rollout_retain', 5))
    num_repeat = int(cfg.get('num_repeat', 1))
    actor_update = str(cfg.get('actor_update', 'lambda-return'))
    critic_update = str(cfg.get('critic_update', 'lambda-return'))
    video_interval = int(cfg.get('video_interval', 0))
    eval_interval = int(cfg.get('eval_interval', 50000))
    save_interval = int(cfg.get('save_interval', 100000))

    wandb_cfg = cfg.get('wandb') or {}
    wandb_project = args.project or wandb_cfg.get(
        'project', 'leq_mountaincar_hyperparameter_search'
    )
    track = not args.no_track
    save_root = Path(args.save_root or cfg.get('save_root') or DEFAULT_SAVE_ROOT)
    if not save_root.is_absolute():
        save_root = LEQ_ROOT / save_root

    dynamics_by_mode: dict[str, dict[str, Any]] = {}
    for reward_mode in reward_mode_list:
        dynamics_by_mode[reward_mode] = load_dynamics_checkpoint(
            reward_mode=reward_mode,
            dynamics_best_run_stem=args.dynamics_best_run_stem,
        )
        info = dynamics_by_mode[reward_mode]
        print(
            f'Dynamics [{reward_mode}]: {info["dynamics_model_name"]} '
            f'(seed={info["dynamics_seed"]}, '
            f'holdout_loss={info.get("holdout_loss")}, '
            f'load_dir={info["load_dir"]})'
        )

    combos = list(
        itertools.product(
            seeds,
            expectile_list,
            model_batch_ratio_list,
            rollout_length_list,
            horizon_length_list,
            lamb_list,
            actor_lr_list,
            discount_list,
            batch_size_list,
            pretrain_list,
            reward_mode_list,
        )
    )
    print(
        f'Starting LEQ grid search: prefix={prefix}, '
        f'env_name={env_name}, seeds={seeds}, '
        f'expectile={expectile_list}, '
        f'model_batch_ratio={model_batch_ratio_list}, '
        f'rollout_length={rollout_length_list}, '
        f'horizon_length={horizon_length_list}, '
        f'lamb={lamb_list}, actor_lr={actor_lr_list}, '
        f'discount={discount_list}, batch_size={batch_size_list}, '
        f'pretrain={pretrain_list}, reward_mode={reward_mode_list}, '
        f'max_steps={max_steps}, '
        f'n_combos={len(combos)}, '
        f'project={wandb_project}, track={track}, save_root={save_root}'
    )
    if not combos:
        raise RuntimeError('No search combos to run.')

    best_by_mode: dict[str, dict[str, Any]] = {}
    for (
        seed,
        expectile,
        model_batch_ratio,
        rollout_length,
        horizon_length,
        lamb,
        actor_lr,
        discount,
        batch_size,
        pretrain,
        reward_mode,
    ) in combos:
        dyn = dynamics_by_mode[reward_mode]
        run_name = build_run_name(
            prefix,
            expectile=expectile,
            model_batch_ratio=model_batch_ratio,
            rollout_length=rollout_length,
            horizon_length=horizon_length,
            lamb=lamb,
            actor_lr=actor_lr,
            discount=discount,
            batch_size=batch_size,
            pretrain=pretrain,
            reward_mode=reward_mode,
            seed=seed,
        )
        save_dir = save_root / run_name
        final_score = run_leq(
            env_name=env_name,
            seed=seed,
            expectile=expectile,
            model_batch_ratio=model_batch_ratio,
            rollout_length=rollout_length,
            horizon_length=horizon_length,
            lamb=lamb,
            actor_lr=actor_lr,
            value_lr=value_lr,
            critic_lr=critic_lr,
            discount=discount,
            batch_size=batch_size,
            max_steps=max_steps,
            num_layers=num_layers,
            layer_size=layer_size,
            rollout_batch_size=rollout_batch_size,
            rollout_freq=rollout_freq,
            rollout_retain=rollout_retain,
            num_repeat=num_repeat,
            actor_update=actor_update,
            critic_update=critic_update,
            pretrain=pretrain,
            video_interval=video_interval,
            eval_interval=eval_interval,
            save_interval=save_interval,
            load_dir=dyn['load_dir'],
            run_name=run_name,
            save_dir=save_dir,
            wandb_project=wandb_project,
            track=track,
            wandb_key=args.wandb_key,
            extra_args=args.overrides,
        )
        if final_score is None:
            print(
                f'Warning: could not parse final score for {run_name}; '
                'skipping for best-run selection.'
            )
            continue

        print(f'Parsed final score for {run_name}: {final_score}')
        candidate = {
            'env_name': env_name,
            'task': env_name,
            'seed': seed,
            'output_model_name': run_name,
            'save_dir': str(save_dir),
            'expectile': expectile,
            'model_batch_ratio': model_batch_ratio,
            'rollout_length': rollout_length,
            'horizon_length': horizon_length,
            'lamb': lamb,
            'actor_lr': actor_lr,
            'value_lr': value_lr,
            'critic_lr': critic_lr,
            'discount': discount,
            'batch_size': batch_size,
            'pretrain': pretrain,
            'reward_mode': reward_mode,
            'dynamics_seed': dyn['dynamics_seed'],
            'dynamics_model_name': dyn['dynamics_model_name'],
            'dynamics_load_dir': dyn['load_dir'],
            'dynamics_best_run_path': dyn['best_run_path'],
            'dynamics_holdout_loss': dyn.get('holdout_loss'),
            'max_steps': max_steps,
            'num_layers': num_layers,
            'layer_size': layer_size,
            'rollout_batch_size': rollout_batch_size,
            'rollout_freq': rollout_freq,
            'rollout_retain': rollout_retain,
            'num_repeat': num_repeat,
            'actor_update': actor_update,
            'critic_update': critic_update,
            'final_score': final_score,
            'wandb': {
                'project': wandb_project,
                'name': run_name,
            },
        }
        prev = best_by_mode.get(reward_mode)
        if prev is None or final_score > float(prev['final_score']):
            best_by_mode[reward_mode] = candidate
            print(
                f'New best {reward_mode} run: {run_name} '
                f'(final_score={final_score})'
            )

    if not best_by_mode:
        raise RuntimeError(
            'No successful runs with a parseable final score; '
            f'not writing best-run YAMLs under {args.best_run_path}'
        )

    missing_modes = [m for m in reward_mode_list if m not in best_by_mode]
    if missing_modes:
        print(
            'Warning: no parseable final score for reward mode(s): '
            f'{missing_modes}'
        )

    for reward_mode, payload in best_by_mode.items():
        out_path = best_run_path_for_mode(args.best_run_path, reward_mode)
        save_best_run(out_path, payload)
        print(
            f'Best {reward_mode}: {payload["output_model_name"]} '
            f'(final_score={payload["final_score"]})'
        )


if __name__ == '__main__':
    main()
