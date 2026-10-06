"""Atomic training checkpoints and deterministic batch order across Slurm jobs."""
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

PAUSED_EXIT_CODE = 75
_stop_requested = False


def request_stop(*_):
    global _stop_requested
    _stop_requested = True


def install_checkpoint_signals():
    global _stop_requested
    _stop_requested = False
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def pause_if_requested():
    # Called during validation only after a training checkpoint has been committed.
    if _stop_requested:
        raise SystemExit(PAUSED_EXIT_CODE)


def add_resume_arguments(parser):
    parser.add_argument('--checkpoint_interval_seconds', type=float, default=600,
                        help='Periodic checkpoint interval at optimizer boundaries (default: 600s)')


def epoch_loader(dataset, batch_size, collate_fn, workers, seed, epoch, start_batch=0):
    # A dedicated generator keeps loader construction from consuming model/dropout RNG.
    generator = torch.Generator().manual_seed(seed + epoch)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    if not 0 <= start_batch <= len(batches):
        raise ValueError('Checkpoint batch position is outside the dataset')
    return DataLoader(dataset, batch_sampler=batches[start_batch:], collate_fn=collate_fn,
                      num_workers=workers, generator=generator,
                      pin_memory=torch.cuda.is_available())


class TrainingResume:
    def __init__(self, path, model, optimizer, scheduler, scaler=None, interval=600):
        if interval <= 0:
            raise ValueError('Checkpoint interval must be positive')
        self.path = Path(path)
        self.model, self.optimizer, self.scheduler, self.scaler = model, optimizer, scheduler, scaler
        self.interval = interval
        self.last_save = time.monotonic()

    def load(self):
        if not self.path.exists():
            return None
        checkpoint = torch.load(self.path, map_location='cpu', weights_only=False)
        if checkpoint.get('resume_format') != 1:
            raise ValueError('Legacy checkpoint is not compatible with batch-level resumption; use a new study directory')
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.scaler is not None and checkpoint['scaler'] is not None:
            self.scaler.load_state_dict(checkpoint['scaler'])
        random.setstate(checkpoint['rng']['python'])
        np.random.set_state(checkpoint['rng']['numpy'])
        torch.set_rng_state(checkpoint['rng']['torch'])
        if torch.cuda.is_available() and checkpoint['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
        print(f'Resuming {self.path}: epoch={checkpoint["progress"]["next_epoch"]}, '
              f'batch={checkpoint["progress"]["next_batch"]}', flush=True)
        return checkpoint['progress']

    def save(self, progress):
        checkpoint = {
            'resume_format': 1, 'progress': progress,
            'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler is not None else None,
            'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.pt.tmp')
        with temporary.open('wb') as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        self.last_save = time.monotonic()
        print(f'Checkpoint committed: {self.path} '
              f'(epoch={progress["next_epoch"]}, batch={progress["next_batch"]})', flush=True)

    def maybe_save(self, progress, force=False):
        if force or _stop_requested or time.monotonic() - self.last_save >= self.interval:
            self.save(progress)
        pause_if_requested()
