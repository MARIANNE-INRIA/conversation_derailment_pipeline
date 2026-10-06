#!/usr/bin/env python3
"""Join annot1/annot2 onto test prefixes using conversation and message position only."""
import argparse
import csv
import json
from pathlib import Path
import re

csv.field_size_limit(64 * 1024 * 1024)


def annotate(samples, source_dir, output):
    if samples.resolve() == output.resolve():
        raise ValueError('Use a separate output file to preserve the original test samples')
    with samples.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)
    ids = {row['conversation_id'] for row in rows}
    lookup, skipped = {}, []
    files = sorted(source_dir.glob('*.csv'))
    if not files:
        raise ValueError(f'No annotation CSVs in {source_dir}')
    for path in files:
        match = re.fullmatch(r'(.+)_(?:NOT_TOXIC|TOXIC)_annotated\.csv', path.name)
        if not match or match[1] not in ids:
            continue
        # The exported ID cells lost punctuation/precision; filenames preserve IDs.
        cid = match[1]
        with path.open(newline='', encoding='utf-8-sig') as handle:
            for line, row in enumerate(csv.DictReader(handle), start=2):
                turn = row['turn_index'].strip()
                if not re.fullmatch(r'\d+', turn):
                    skipped.append(dict(file=path.name, record=line, turn_index=turn))
                    continue
                key = (cid, int(turn) + 1)
                values = (row['annot1'], row['annot2'])
                if key in lookup and lookup[key] != values:
                    raise ValueError(f'Conflicting annotations for {key}')
                lookup[key] = values
    matched, matched_ids = 0, set()
    for row in rows:
        key = (row['conversation_id'], int(row['timestep']))
        row['annot1'], row['annot2'] = lookup.get(key, ('', ''))
        if key in lookup:
            matched += 1
            matched_ids.add(key[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(columns + ['annot1', 'annot2'])))
        writer.writeheader()
        writer.writerows(rows)
    audit = dict(samples=str(samples), source_dir=str(source_dir), output=str(output),
                 rows=len(rows), matched_rows=matched, unmatched_rows=len(rows)-matched,
                 matched_conversations=len(matched_ids), skipped_source_rows=skipped,
                 join='conversation_id from filename; timestep = turn_index + 1; no text filtering',
                 scope='annot1/annot2 for the message at timestep, not an aggregation of the prefix')
    output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=Path('data/test_samplesLLMs.csv'))
    parser.add_argument('--source_dir', type=Path, default=Path('pragmaticsAggressionRedditConversations/CGA-Wiki Pragmatics annotation'))
    parser.add_argument('--output', type=Path, default=Path('data/test_samplesLLMs_annotated.csv'))
    args = parser.parse_args()
    print(json.dumps(annotate(args.samples, args.source_dir, args.output), indent=2))


if __name__ == '__main__':
    main()
