#!/usr/bin/env python3
"""Rank complete multi-seed confirmations; freeze the winner before test evaluation."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from hpo_common import write_json


def summarize(output):
    output = Path(output).resolve()
    manifest = json.loads((output / 'confirmation.json').read_text())
    rows = []
    for number in manifest['trials']:
        runs = [output / 'confirmation' / f'trial_{number:04d}' / f'seed_{seed}' for seed in manifest['seeds']]
        if any(not (run / 'result.json').exists() for run in runs):
            continue
        results = [json.loads((run / 'result.json').read_text()) for run in runs]
        if any(result['status'] != 'complete' for result in results):
            continue
        scores = [result['conv_f1'] for result in results]
        rows.append({'trial': number, 'mean_conv_f1': statistics.mean(scores),
                     'std_conv_f1': statistics.stdev(scores) if len(scores) > 1 else 0,
                     'n_seeds': len(scores), 'duration_seconds': sum(r['duration_seconds'] for r in results)})
    rows.sort(key=lambda row: (-row['mean_conv_f1'], row['std_conv_f1'], row['trial']))
    with (output / 'leaderboard.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['trial', 'mean_conv_f1', 'std_conv_f1', 'n_seeds', 'duration_seconds'])
        writer.writeheader()
        writer.writerows(rows)
    if len(rows) != len(manifest['trials']):
        raise ValueError('Confirmation incomplete: leaderboard is partial, no winner selected')
    winner = rows[0]
    runs = [str(output / 'confirmation' / f"trial_{winner['trial']:04d}" / f'seed_{seed}') for seed in manifest['seeds']]
    write_json(output / 'selected.json', {**winner, 'runs': runs, 'seeds': manifest['seeds']})
    print(json.dumps(winner, indent=2))
    return winner


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', required=True)
    summarize(parser.parse_args().output_dir)
