#!/usr/bin/env python3
"""Sequential, restartable Optuna search and multi-seed confirmation on one GPU."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from hpo_common import EVENT_PREFIX, check_splits, fingerprint, write_json

ROOT = Path(__file__).resolve().parents[1]
PAUSED_EXIT_CODE = 75
_pause_requested = False
_active_process = None


class TrainingPaused(Exception):
    """Keep the Optuna trial RUNNING so its identity survives job resubmission."""


def request_pause(*_):
    global _pause_requested
    _pause_requested = True
    if _active_process is not None and _active_process.poll() is None:
        try:
            # Only the trainer handles this signal, not DataLoader worker processes.
            os.kill(_active_process.pid, signal.SIGUSR1)
        except ProcessLookupError:
            pass


def install_pause_signals():
    global _pause_requested
    _pause_requested = False
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, request_pause)



def training_command(config, parameters, seed, output):
    options = {**config['fixed'], **parameters}
    effective = options.pop('effective_batch_size')
    if config['trainer'] == 'llm':
        micro = options['batch_size']
        if effective < micro or effective % micro:
            raise ValueError('Effective batch size must be divisible by micro-batch size')
        options['gradient_accumulation_steps'] = effective // micro
        options['lora_alpha'] = 2 * options['lora_r']
    else:
        options['batch_size'] = effective
    options.update(model_name=config['model_name'], train_path=config['train_path'],
                   val_path=config['val_path'], seed=seed, output_dir=str(output))
    script = 'train_llm_forecasting.py' if config['trainer'] == 'llm' else f"train_{config['trainer']}.py"
    command = [sys.executable, '-u', str(ROOT / 'scripts' / script)]
    for key, value in options.items():
        if value is True:
            command.append('--' + key)
        elif value is not False and value is not None:
            command.extend(['--' + key, str(value)])
    return command


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_training(config, parameters, seed, output, trial=None):
    global _active_process
    if _pause_requested:
        raise TrainingPaused('Walltime warning received before starting the next training run')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    command = training_command(config, parameters, seed, output)
    spec = {'command': command, 'parameters': parameters, 'seed': seed}
    spec_path = output / 'launch.json'
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError(f'Existing run has different settings: {output}')
    write_json(spec_path, spec)
    result_path = output / 'result.json'
    previous_duration = 0.0
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        previous_duration = previous.get('duration_seconds', 0.0)
        if previous['status'] == 'complete':
            return previous['conv_f1']
    started = time.monotonic()
    result = {'status': 'running', 'parameters': parameters, 'seed': seed}
    write_json(result_path, result)
    process = None
    try:
        with (output / 'console.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, start_new_session=True)
            _active_process = process
            if _pause_requested:
                request_pause()
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
                if trial is not None and line.startswith(EVENT_PREFIX):
                    event = json.loads(line[len(EVENT_PREFIX):])
                    if event['epoch'] not in trial.storage.get_trial(trial._trial_id).intermediate_values:
                        trial.report(event['conv_f1'], step=event['epoch'])
                    if trial.should_prune():
                        import optuna
                        result['status'] = 'pruned'
                        raise optuna.TrialPruned()
            returncode = process.wait()
            if returncode == PAUSED_EXIT_CODE or (_pause_requested and returncode != 0):
                raise TrainingPaused('Checkpoint pause; resubmit the same command to continue')
            if returncode != 0:
                raise RuntimeError(f'Training failed; see {output / "console.log"}')
        with (output / 'validation_metrics.csv').open() as handle:
            metrics = next(csv.DictReader(handle))
        score = float(metrics['conv_f1'])
        if not 0 <= score <= 1:
            raise RuntimeError('Invalid final F1')
        result.update(status='complete', conv_f1=score, metrics=metrics,
                      best_tau=json.loads((output / 'best_tau.json').read_text())['best_tau'])
        return score
    except BaseException as error:
        if result['status'] != 'pruned':
            result['status'] = 'interrupted' if isinstance(error, (TrainingPaused, KeyboardInterrupt, SystemExit)) else 'failed'
        result['error'] = str(error)
        raise
    finally:
        if process is not None:
            stop_process(process)
            if process.stdout is not None:
                process.stdout.close()
        _active_process = None
        result['duration_seconds'] = previous_duration + time.monotonic() - started
        write_json(result_path, result)


def sample(trial, space):
    return {key: trial.suggest_categorical(key, values) if isinstance(values, list)
            else trial.suggest_float(key, **values) for key, values in space.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--phase', choices=['baseline', 'search', 'confirm'], default='search')
    parser.add_argument('--trials', type=int, default=20, help='Total attempted trials, including reference trial')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--epochs', type=int, help='Override search/baseline budget; use a new output directory')
    parser.add_argument('--confirm_epochs', type=int, default=5)
    parser.add_argument('--group_column', default='conversation_id')
    parser.add_argument('--max_runtime_seconds', type=float, help='Request a checkpoint pause after this duration; for local tests or an extra time budget')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    if args.max_runtime_seconds is not None and args.max_runtime_seconds <= 0:
        parser.error('--max_runtime_seconds must be positive')
    if args.max_runtime_seconds is not None:
        signal.setitimer(signal.ITIMER_REAL, args.max_runtime_seconds)
    if min(args.trials, args.top_k, args.confirm_epochs, args.epochs or 1) < 1:
        parser.error('Budgets must be positive')
    config = json.loads(Path(args.config).read_text())
    if args.epochs:
        config['fixed']['epochs'] = args.epochs
    for key in ('train_path', 'val_path'):
        config[key] = str((ROOT / config[key]).resolve())
    output = Path(args.output_dir).resolve()
    if args.dry_run:
        print(json.dumps(training_command(config, config['baseline'], args.seed, output / 'baseline'), indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    # SQLite study is intentionally restricted to one orchestrator at a time.
    with (output / '.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another orchestrator is using this output directory')
        audit = check_splits({'train': config['train_path'], 'validation': config['val_path']}, args.group_column)
        protocol = {'config': config, 'seed': args.seed, 'data': audit, 'group_column': args.group_column,
                    'code': {p.name: fingerprint(p) for p in [Path(__file__), ROOT / 'scripts/hpo_common.py', ROOT / 'scripts/online_alerts.py', ROOT / 'scripts/training_resume.py',
                              ROOT / 'scripts' / ('train_llm_forecasting.py' if config['trainer'] == 'llm' else f"train_{config['trainer']}.py")]}}
        protocol_path = output / 'protocol.json'
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            parser.error('Data, code or configuration changed; use a new output directory')
        write_json(protocol_path, protocol)
        if args.phase == 'baseline':
            run_training(config, config['baseline'], args.seed, output / 'baseline')
            return
        import optuna
        study = optuna.create_study(study_name='hpo', storage='sqlite:///' + str(output / 'study.sqlite3'),
                                   direction='maximize', load_if_exists=True,
                                   sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=8),
                                   pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2))
        if args.phase == 'search':
            if not study.trials:
                study.enqueue_trial(config['baseline'])
            while True:
                if _pause_requested:
                    raise TrainingPaused('Walltime warning; no new trial will be started')
                running = study.get_trials(states=(optuna.trial.TrialState.RUNNING,))
                if len(running) > 1:
                    raise RuntimeError('Multiple RUNNING trials in a sequential study; inspect the database')
                if running:
                    # Optuna exposes the Trial(study, trial_id) constructor for attaching
                    # to an existing trial. FrozenTrial stores its database ID here.
                    trial = optuna.trial.Trial(study, running[0]._trial_id)
                else:
                    attempted = sum(t.state != optuna.trial.TrialState.WAITING for t in study.trials)
                    if attempted >= args.trials:
                        break
                    trial = study.ask()
                try:
                    parameters = sample(trial, config['space'])
                    score = run_training(config, parameters, args.seed, output / f'trial_{trial.number:04d}', trial)
                except (TrainingPaused, KeyboardInterrupt, SystemExit):
                    # Crucially, do not tell Optuna FAIL/PRUNED on a walltime pause.
                    raise
                except optuna.TrialPruned:
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                except Exception:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    raise
                else:
                    study.tell(trial, score)
            study.trials_dataframe().to_csv(output / 'trials.csv', index=False)
        else:
            completed = sorted(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)), key=lambda t: (-t.value, t.number))
            unique = []
            for trial in completed:
                if trial.params not in [t.params for t in unique]:
                    unique.append(trial)
            if len(unique) < args.top_k:
                parser.error(f'Need {args.top_k} distinct completed trials; found {len(unique)}')
            manifest = {'trials': [t.number for t in unique[:args.top_k]], 'seeds': list(dict.fromkeys(args.seeds)),
                        'epochs': args.confirm_epochs}
            manifest_path = output / 'confirmation.json'
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
                parser.error('Confirmation selection/budget changed; keep the original study and confirmation settings')
            write_json(manifest_path, manifest)
            config['fixed']['epochs'] = args.confirm_epochs
            for trial in unique[:args.top_k]:
                for seed in manifest['seeds']:
                    run_training(config, trial.params, seed,
                                 output / 'confirmation' / f'trial_{trial.number:04d}' / f'seed_{seed}')
            from summarize_hpo import summarize
            summarize(output)


if __name__ == '__main__':
    install_pause_signals()
    try:
        main()
    except TrainingPaused as error:
        print(str(error), flush=True)
        sys.exit(PAUSED_EXIT_CODE)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

