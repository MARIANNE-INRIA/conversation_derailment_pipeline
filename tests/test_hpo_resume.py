"""Exercise checkpoint parity, real process signals, and Optuna trial identity."""
import os
os.environ.setdefault('USE_TF', '0')
import json
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
import torch
import run_hpo
import training_resume
from hpo_common import write_json
from training_resume import TrainingResume, epoch_loader
from train_llm_forecasting import train_one_epoch


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(.3)
        self.layer = torch.nn.Linear(2, 2)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.layer(self.dropout(input_ids.float())))


def make_components(path):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - .05 * step)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    return model, optimizer, scheduler, TrainingResume(path, model, optimizer, scheduler, scaler)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ResumeTests(unittest.TestCase):
    def tearDown(self):
        training_resume._stop_requested = False
        run_hpo._pause_requested = False

    def test_mid_epoch_resume_matches_uninterrupted_with_dropout(self):
        dataset = [{'input_ids':torch.tensor([i / 10, 1.]), 'attention_mask':torch.ones(2),
                    'labels':torch.tensor(i % 2)} for i in range(9)]
        kwargs = dict(device=torch.device('cpu'), epoch=0, total_epochs=1, log_every=100,
                      label_smoothing=0, gradient_accumulation_steps=2, amp_dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'state.pt'
            seed_all(123)
            full, full_opt, full_sched, _ = make_components(checkpoint)
            expected_loss = train_one_epoch(full, epoch_loader(dataset, 2, None, 0, 42, 0), full_opt, full_sched, **kwargs)
            expected_random = (random.random(), np.random.rand(), torch.rand(2))
            seed_all(123)
            interrupted, opt, sched, resume = make_components(checkpoint)
            def stop(step, loss):
                if step == 2:
                    training_resume.request_stop()
                resume.maybe_save({'next_epoch':0, 'next_batch':step, 'loss_sum':loss})
            with self.assertRaises(SystemExit) as caught:
                train_one_epoch(interrupted, epoch_loader(dataset, 2, None, 0, 42, 0), opt, sched,
                                checkpoint_callback=stop, **kwargs)
            self.assertEqual(caught.exception.code, 75)
            training_resume._stop_requested = False
            seed_all(999)
            resumed, opt, sched, resume = make_components(checkpoint)
            progress = resume.load()
            actual_loss = train_one_epoch(resumed, epoch_loader(dataset, 2, None, 0, 42, 0, progress['next_batch']),
                                          opt, sched, start_batch=progress['next_batch'], loss_sum=progress['loss_sum'], **kwargs)
            for key, value in full.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(full_sched.state_dict(), sched.state_dict())
            self.assertEqual(expected_loss, actual_loss)
            self.assertEqual(random.random(), expected_random[0])
            self.assertEqual(np.random.rand(), expected_random[1])
            torch.testing.assert_close(torch.rand(2), expected_random[2], rtol=0, atol=0)

    def test_failed_atomic_write_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, resume = make_components(Path(tmp) / 'state.pt')
            resume.save({'next_epoch':0, 'next_batch':2})
            with patch('training_resume.torch.save', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    resume.save({'next_epoch':0, 'next_batch':4})
            self.assertEqual(resume.load()['next_batch'], 2)

    def test_real_child_signal_pause_then_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            code = '''import pathlib,signal,sys,time
p=pathlib.Path(sys.argv[1])
if (p/'saved').exists():
    (p/'validation_metrics.csv').write_text('conv_f1\\n0.8\\n')
    (p/'best_tau.json').write_text('{"best_tau":0.3}')
else:
    def pause(*_):
        (p/'saved').write_text('checkpoint')
        raise SystemExit(75)
    signal.signal(signal.SIGUSR1,pause)
    (p/'ready').touch()
    time.sleep(20)
'''
            def send_signal():
                for _ in range(500):
                    if (output/'ready').exists():
                        run_hpo.request_pause()
                        return
                    time.sleep(.01)
            thread = threading.Thread(target=send_signal)
            with patch('run_hpo.training_command', return_value=[sys.executable, '-c', code, str(output)]):
                thread.start()
                with self.assertRaises(run_hpo.TrainingPaused):
                    run_hpo.run_training({}, {}, 42, output)
                thread.join(timeout=6)
                first = json.loads((output/'result.json').read_text())
                self.assertEqual(first['status'], 'interrupted')
                run_hpo._pause_requested = False
                self.assertEqual(run_hpo.run_training({}, {}, 42, output), .8)
                self.assertGreaterEqual(json.loads((output/'result.json').read_text())['duration_seconds'], first['duration_seconds'])

    def test_slurm_wrapper_waits_for_checkpoint_before_requeue(self):
        root_repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'etc/profile.d').mkdir(parents=True)
            (root/'etc/profile.d/conda.sh').write_text('conda() { export CONDA_PREFIX="$HPO_CONDA_PREFIX"; }\n')
            (root/'env/bin').mkdir(parents=True)
            scripts = {
                'module': '#!/bin/bash\nexit 0\n',
                'conda': '#!/bin/bash\nprintf "%s\\n" "$MOCK_ROOT"\n',
                'scontrol': '#!/bin/bash\ntest -f "$MOCK_ROOT/saved" || exit 9\nprintf "%s\\n" "$*" > "$MOCK_ROOT/requeued"\n',
                'python': '#!/bin/bash\nif [[ "$1" == -c ]]; then exit 0; fi\n'
                          'trap \'sleep 0.2; touch "$MOCK_ROOT/saved"; exit 75\' USR1\n'
                          'touch "$MOCK_ROOT/ready"\nwhile true; do sleep 0.05; done\n',
            }
            for name, code in scripts.items():
                path = root/'env/bin/python' if name == 'python' else root/name
                path.write_text(code)
                path.chmod(0o755)
            # A misleading Python on PATH must never be used by the job.
            (root/'python').write_text('#!/bin/bash\nexit 92\n')
            (root/'python').chmod(0o755)
            environment = {**os.environ, 'PATH':str(root)+os.pathsep+os.environ['PATH'],
                           'MOCK_ROOT':str(root), 'SLURM_SUBMIT_DIR':str(root_repo),
                           'HPO_CONDA_PREFIX':str(root/'env'),
                           'SLURM_JOB_ID':'123', 'HPO_AUTO_REQUEUE':'1', 'SLURM_RESTART_COUNT':'0'}
            with subprocess.Popen(['bash', str(root_repo/'jobs/submit_hpo.sh')], env=environment,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
                for _ in range(500):
                    if (root/'ready').exists():
                        break
                    time.sleep(.01)
                self.assertTrue((root/'ready').exists())
                process.send_signal(signal.SIGUSR1)
                output, _ = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 75, output)
                self.assertEqual((root/'requeued').read_text().strip(), 'requeue 123')

    def test_encoder_training_restarts_from_saved_batch(self):
        import importlib
        import contextlib
        import io
        class Encoder(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.layer = torch.nn.Linear(1, 2)
                self.dropout = torch.nn.Dropout(.2)
            def forward(self, input_ids, attention_mask, labels=None):
                logits = self.layer(self.dropout(input_ids.float()))
                loss = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else None
                return loss, logits
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = {'input_ids':[torch.tensor([i]) for i in range(8)],
                    'attention_mask':[torch.ones(1, dtype=torch.long) for _ in range(8)],
                    'labels':[i%2 for i in range(8)], 'conversation_ids':[str(i) for i in range(8)],
                    'timesteps':[1]*8,'total_turns':[2]*8}
            torch.save(data, root/'data.pt')
            original_save = TrainingResume.maybe_save
            def pause(self, progress, force=False):
                if progress['next_epoch'] == 1 and progress['next_batch'] == 2:
                    training_resume.request_stop()
                original_save(self, progress, force)
            for family, cls in [('roberta','RoBERTaForDerailment'), ('deberta','DeBERTaForDerailment')]:
                module = importlib.import_module('train_'+family)
                def train(output):
                    argv = ['train', '--model_name','fake','--train_path',str(root/'data.pt'),
                            '--val_path',str(root/'data.pt'),'--output_dir',str(output),
                            '--epochs','2','--batch_size','2','--workers','0']
                    with patch.object(module, cls, Encoder), patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                        module.main()
                full = root/(family+'_full')
                restarted = root/(family+'_restarted')
                train(full)
                with patch.object(TrainingResume, 'maybe_save', pause):
                    with self.assertRaises(SystemExit) as caught:
                        train(restarted)
                    self.assertEqual(caught.exception.code, 75)
                train(restarted)
                a = torch.load(full/'training_resume.pt', weights_only=False)
                b = torch.load(restarted/'training_resume.pt', weights_only=False)
                for key in a['model']:
                    torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
                self.assertEqual(a['scheduler'], b['scheduler'])
                self.assertEqual(a['progress'], b['progress'])

    def test_walltime_keeps_same_optuna_trial_and_budget(self):
        try:
            import optuna
        except ImportError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['train','val']:
                (root / (name+'.csv')).write_text(f'conversation_id,label\n{name}0,0\n{name}1,1\n')
            config = {'trainer':'llm', 'model_name':'fake', 'fixed':{}, 'baseline':{'lr':.1},
                      'space':{'lr':{'low':.1,'high':.9}}, 'train_path':str(root/'train.csv'), 'val_path':str(root/'val.csv')}
            write_json(root/'config.json', config)
            argv = ['run_hpo', '--config', str(root/'config.json'), '--output_dir', str(root/'study'), '--trials','1']
            def pause(config, parameters, seed, output, trial):
                trial.report(.4, step=1)
                raise run_hpo.TrainingPaused()
            with patch('sys.argv', argv), patch('run_hpo.run_training', side_effect=pause):
                with self.assertRaises(run_hpo.TrainingPaused):
                    run_hpo.main()
            study = optuna.load_study(study_name='hpo', storage='sqlite:///' + str(root/'study/study.sqlite3'))
            self.assertEqual(study.trials[0].state, optuna.trial.TrialState.RUNNING)
            def finish(config, parameters, seed, output, trial):
                self.assertEqual(trial.number, 0)
                self.assertEqual(parameters, {'lr':.1})
                self.assertEqual(trial.storage.get_trial(trial._trial_id).intermediate_values, {1:.4})
                return .8
            with patch('sys.argv', argv), patch('run_hpo.run_training', side_effect=finish):
                run_hpo.main()
            self.assertEqual(len(study.trials), 1)
            self.assertEqual(study.trials[0].state, optuna.trial.TrialState.COMPLETE)
            self.assertEqual(study.trials[0].value, .8)


if __name__ == '__main__':
    unittest.main()
