import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from hpo_common import check_splits, threshold_grid, write_json
from run_hpo import run_training, training_command
from summarize_hpo import summarize


class PipelineTests(unittest.TestCase):
    def test_group_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = {}
            for split in ('train', 'validation'):
                p = Path(tmp) / (split + '.csv')
                p.write_text('conversation_id,label\nshared,0\nother,1\n')
                files[split] = p
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                check_splits(files)

    def test_threshold_grid_includes_endpoints(self):
        grid = threshold_grid(.05, .95, .01)
        self.assertEqual((len(grid), grid[0], grid[-1]), (91, .05, .95))
        with self.assertRaises(ValueError):
            threshold_grid(.05, .95, 0)

    def test_llm_effective_batch(self):
        config = {'trainer': 'llm', 'model_name': 'fake', 'train_path': 'a', 'val_path': 'b',
                  'fixed': {'batch_size': 4}}
        command = training_command(config, {'effective_batch_size': 32, 'lora_r': 16}, 42, '/tmp/run')
        self.assertEqual(command[command.index('--gradient_accumulation_steps') + 1], '8')
        self.assertEqual(command[command.index('--lora_alpha') + 1], '32')

    def test_subprocess_success_and_cached_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            code = '''import pathlib,sys,json
p=pathlib.Path(sys.argv[1])
print('HPO_EPOCH '+json.dumps({'epoch':1,'conv_f1':.8}),flush=True)
(p/'validation_metrics.csv').write_text('conv_f1\\n0.8\\n')
(p/'best_tau.json').write_text('{"best_tau":0.3}')
'''
            command = [sys.executable, '-c', code, str(output)]
            with patch('run_hpo.training_command', return_value=command):
                self.assertEqual(run_training({}, {}, 42, output), .8)
                with patch('run_hpo.subprocess.Popen', side_effect=AssertionError('Must reuse complete run')):
                    self.assertEqual(run_training({}, {}, 42, output), .8)
            self.assertEqual(json.loads((output / 'result.json').read_text())['status'], 'complete')

    def test_failure_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('run_hpo.training_command', return_value=[sys.executable, '-c', 'raise SystemExit(2)']):
                with self.assertRaises(RuntimeError):
                    run_training({}, {}, 42, tmp)
            self.assertEqual(json.loads((Path(tmp) / 'result.json').read_text())['status'], 'failed')

    def test_confirmation_ranks_mean_not_best_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / 'confirmation.json', {'trials': [0, 1], 'seeds': [42, 123, 456]})
            for trial, scores in [(0, [.99, .5, .5]), (1, [.8, .81, .79])]:
                for seed, score in zip([42, 123, 456], scores):
                    p = root / 'confirmation' / f'trial_{trial:04d}' / f'seed_{seed}'
                    p.mkdir(parents=True)
                    write_json(p / 'result.json', {'status':'complete', 'conv_f1':score, 'duration_seconds':1})
            self.assertEqual(summarize(root)['trial'], 1)
            selected = json.loads((root / 'selected.json').read_text())
            self.assertEqual(len(selected['runs']), 3)

    def test_search_confirmation_and_restart_with_sqlite(self):
        try:
            import optuna
        except ImportError as error:
            self.skipTest(str(error))
        import run_hpo
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['train', 'val']:
                (root / (name + '.csv')).write_text(f'conversation_id,label\n{name}_0,0\n{name}_1,1\n')
            config = {'trainer':'llm', 'model_name':'fake',
                      'train_path':str(root/'train.csv'), 'val_path':str(root/'val.csv'),
                      'fixed':{'epochs':5}, 'baseline':{'lr':.1},
                      'space':{'lr':{'low':.1,'high':.9}}}
            write_json(root/'config.json', config)
            def fake_command(config, parameters, seed, output):
                code = ("import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
                        "(p/'validation_metrics.csv').write_text('conv_f1\\n0.8\\n'); "
                        "(p/'best_tau.json').write_text('{\"best_tau\":0.3}')")
                return [sys.executable, '-c', code, str(output)]
            base = ['run_hpo', '--config', str(root/'config.json'), '--output_dir', str(root/'study')]
            with patch('run_hpo.training_command', side_effect=fake_command):
                for phase in ['search', 'search', 'confirm', 'confirm']:
                    with patch('sys.argv', base + ['--phase', phase, '--trials','3','--top_k','2']):
                        run_hpo.main()
            selected = json.loads((root/'study/selected.json').read_text())
            self.assertEqual(selected['n_seeds'], 3)
            study = optuna.load_study(study_name='hpo', storage='sqlite:///' + str(root/'study/study.sqlite3'))
            self.assertEqual(len(study.trials), 3)

    def test_optuna_prunes_and_stops_child(self):
        try:
            import optuna
        except ImportError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as tmp:
            study = optuna.create_study(direction='maximize', pruner=optuna.pruners.ThresholdPruner(lower=.5))
            def objective(trial):
                code = "import time; print('HPO_EPOCH {\"epoch\":2,\"conv_f1\":0.1}',flush=True); time.sleep(30)"
                with patch('run_hpo.training_command', return_value=[sys.executable, '-c', code]):
                    return run_training({}, {}, 42, tmp, trial)
            study.optimize(objective, n_trials=1)
            self.assertEqual(study.trials[0].state, optuna.trial.TrialState.PRUNED)
            self.assertEqual(json.loads((Path(tmp) / 'result.json').read_text())['status'], 'pruned')


if __name__ == '__main__':
    unittest.main()
