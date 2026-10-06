"""Shared HPO protocol, without training or Optuna dependencies."""
import csv
import hashlib
import json
import math
from pathlib import Path

csv.field_size_limit(100_000_000)

EVENT_PREFIX = 'HPO_EPOCH '


def threshold_grid(start, end, step):
    if not all(math.isfinite(x) for x in (start, end, step)) or not 0 <= start <= end <= 1 or step <= 0:
        raise ValueError('Expected 0 <= tau_start <= tau_end <= 1 and tau_step > 0')
    return [round(start + i * step, 10) for i in range(int((end - start) / step + 1e-8) + 1)]


def report_epoch(epoch, score):
    if not math.isfinite(float(score)):
        raise ValueError('Non-finite validation score')
    print(EVENT_PREFIX + json.dumps({'epoch': int(epoch), 'conv_f1': float(score)}), flush=True)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def split_ids(path, group_column='conversation_id'):
    path = Path(path)
    if path.suffix == '.pt':
        if group_column != 'conversation_id':
            raise ValueError('Parent-thread checks require CSV metadata before preparing encoder data')
        import torch
        data = torch.load(path, map_location='cpu', weights_only=False)
        ids, labels = data['conversation_ids'], data['labels']
    else:
        ids, labels = [], []
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if group_column not in (reader.fieldnames or []) or 'label' not in reader.fieldnames:
                raise ValueError(f'Missing group or label column in {path}')
            for row in reader:
                ids.append(row[group_column])
                labels.append(row['label'])
    if not len(ids):
        raise ValueError(f'Empty split: {path}')
    if {int(label) for label in labels} != {0, 1}:
        raise ValueError(f'Expected both binary classes in {path}')
    if any(value is None or str(value).strip() == '' for value in ids):
        raise ValueError(f'Missing group ID in {path}')
    if group_column == 'conversation_id':
        conversation_labels = {}
        for identifier, label in zip(ids, labels):
            key, label = str(identifier), int(label)
            if key in conversation_labels and conversation_labels[key] != label:
                raise ValueError(f'Conflicting labels for conversation {key} in {path}')
            conversation_labels[key] = label
    return {str(value) for value in ids}


def check_splits(paths, group_column='conversation_id'):
    groups = {name: split_ids(path, group_column) for name, path in paths.items()}
    names = list(groups)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            overlap = groups[name] & groups[other]
            if overlap:
                raise ValueError(f'{name}/{other}: {len(overlap)} overlapping {group_column} values')
    return {name: {'groups': len(groups[name]), 'sha256': fingerprint(path)} for name, path in paths.items()}
