#!/usr/bin/env python3
"""Evaluate all seeds of the frozen winning configuration using validation thresholds."""
import argparse
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from hpo_common import check_splits, fingerprint, write_json


def evaluate_run(run, test_path):
    import torch
    from online_alerts import conversation_alert_metrics, replay_alerts
    run = Path(run)
    cfg = json.loads((run / 'run_config.json').read_text())
    tau = json.loads((run / 'best_tau.json').read_text())['best_tau']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = run / 'test_evaluation'
    target.mkdir(exist_ok=True)
    identity = {'test_sha256': fingerprint(test_path), 'tau': tau, 'config': cfg}
    existing = target / 'evaluation_protocol.json'
    if existing.exists() and json.loads(existing.read_text()) != identity:
        raise ValueError('Test inputs or threshold changed for an existing evaluation')
    write_json(existing, identity)
    if (target / 'metrics.json').exists():
        return
    if 'lora_r' in cfg:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel
        from torch.utils.data import DataLoader
        from train_llm_forecasting import (TemporalDataset, TemporalDataCollator, validate,
                                           aggregate_predictions, conversation_metrics, mean_horizon, prefix_metrics)
        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(cfg['amp_dtype'])
        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(run / 'best_model')
        tokenizer.truncation_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
        base = AutoModelForSequenceClassification.from_pretrained(
            cfg['model_name'], num_labels=2, torch_dtype=model_dtype, device_map='auto'
        )
        base.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base, run / 'best_model')
        loader = DataLoader(TemporalDataset(test_path, tokenizer, cfg['max_length']),
                            batch_size=cfg['batch_size'], collate_fn=TemporalDataCollator(tokenizer))
        _, predictions = validate(model, loader, device, amp_dtype=dtype)
        conversations = aggregate_predictions(predictions, tau)
        metrics = {'conv_' + k: v for k, v in conversation_metrics(conversations).items()}
        metrics.update({'prefix_' + k: v for k, v in prefix_metrics(predictions, tau).items()})
        metrics['mean_H'] = mean_horizon(predictions, tau)
    else:
        family = 'deberta' if 'label_smoothing' in cfg else 'roberta'
        module = importlib.import_module('train_' + family)
        checkpoint = torch.load(run / 'best_model.pt', map_location='cpu', weights_only=False)
        cls = module.DeBERTaForDerailment if family == 'deberta' else module.RoBERTaForDerailment
        model = cls(cfg['model_name'], cfg['dropout'])
        # Class weights affect training loss only; the checkpoint may contain the buffer.
        state = checkpoint['model_state']
        if 'class_weights' in state and 'class_weights' not in model.state_dict():
            state = {k: v for k, v in state.items() if k != 'class_weights'}
        model.load_state_dict(state)
        model.to(device)
        loader = module.make_loader(module.CGATemporalDataset(test_path), cfg['batch_size'], False, cfg['workers'])
        metrics, _, predictions = module.evaluate(model, loader, device, tau)
        metrics.update(module.prefix_metrics(predictions, tau))
    alert_replay, alert_summary = replay_alerts(predictions, tau)
    metrics.update(conversation_alert_metrics(alert_summary))
    conversations = alert_summary.rename(columns={'alerted': 'prediction', 'first_alert_timestep': 'first_trigger'})
    conversations.to_csv(target / 'conversation_predictions.csv', index=False)
    alert_replay.to_csv(target / 'online_alert_replay.csv', index=False)
    alert_summary.to_csv(target / 'online_alert_conversation_summary.csv', index=False)
    predictions.to_csv(target / 'prefix_predictions.csv', index=False)
    write_json(target / 'metrics.json', {k: float(v) for k, v in metrics.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir')
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--run', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run:
        evaluate_run(args.run, args.test_path)
        return
    if not args.output_dir:
        parser.error('--output_dir is required')
    output = Path(args.output_dir).resolve()
    protocol = json.loads((output / 'protocol.json').read_text())
    cfg = protocol['config']
    audit = check_splits({'train': cfg['train_path'], 'validation': cfg['val_path'], 'test': args.test_path}, protocol['group_column'])
    for split in ('train', 'validation'):
        if audit[split] != protocol['data'][split]:
            raise ValueError('Training/validation data changed after selection')
    selected = json.loads((output / 'selected.json').read_text())
    run_metrics = []
    for run in selected['runs']:
        subprocess.run([sys.executable, __file__, '--run', run, '--test_path', str(Path(args.test_path).resolve())], check=True)
        run_metrics.append(json.loads((Path(run) / 'test_evaluation/metrics.json').read_text()))
    metric_names = sorted({name for metrics in run_metrics for name, value in metrics.items()
                           if isinstance(value, (int, float))})
    report = {'trial': selected['trial'], 'seeds': selected['seeds'],
              'metrics_per_seed': run_metrics, 'test_sha256': audit['test']['sha256']}
    for name in metric_names:
        values = [float(metrics[name]) for metrics in run_metrics if name in metrics]
        report[f'mean_{name}'] = statistics.mean(values)
        report[f'std_{name}'] = statistics.stdev(values) if len(values) > 1 else 0.0
    write_json(output / 'test_summary.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
