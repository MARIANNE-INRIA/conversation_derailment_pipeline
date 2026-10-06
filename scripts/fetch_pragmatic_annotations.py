#!/usr/bin/env python3
"""Fetch and normalize the human CGA-Wiki annotations, preserving source provenance."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import quote
from urllib.request import urlopen

REPO = 'MARIANNE-INRIA/pragmaticsAggressionRedditConversations'
REVISION = '303080fe9ad52642f023541980ed036c6439d36b'
SUBSET = 'CGA-Wiki Pragmatics annotation/'
# Conversation prefixes can contain substantially more than 128 KiB of text.
csv.field_size_limit(64 * 1024 * 1024)
FIELDS = ['conversation_id', 'timestep', 'annot1', 'annot1_comment', 'annot2',
          'annot2_comment', 'author', 'text', 'source_conversation_id',
          'source_turn_index', 'source_file', 'source_revision']


def download(url, path):
    if not path.exists():
        with urlopen(url, timeout=60) as response:
            content = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.part')
        temporary.write_bytes(content)
        temporary.replace(path)
    return path.read_bytes()


def normalize(content, filename, revision):
    match = re.fullmatch(r'(.+)_(NOT_TOXIC|TOXIC)_annotated\.csv', Path(filename).name)
    if not match:
        raise ValueError(f'Unexpected Wiki filename: {filename}')
    rows = list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))))
    if not rows or not {'turn_index', 'text', 'annot1', 'annot2', 'speaker'} <= rows[0].keys():
        raise ValueError(f'Missing Wiki annotation columns: {filename}')
    indices = [int(row['turn_index']) for row in rows]
    if sorted(indices) != list(range(len(rows))):
        raise ValueError(f'Expected unique contiguous zero-based turn indices: {filename}')
    result = []
    for row in sorted(rows, key=lambda r: int(r['turn_index'])):
        result.append(dict(conversation_id=match[1], timestep=str(int(row['turn_index']) + 1),
                           **{key: row.get(key, '') for key in FIELDS[2:6]},
                           author=row['speaker'], text=row['text'],
                           source_conversation_id=row.get('conversation_id', ''),
                           source_turn_index=row['turn_index'], source_file=filename,
                           source_revision=revision))
    return result


def coverage(samples_path, annotations):
    lookup = {(r['conversation_id'], r['timestep']): r for r in annotations}
    conversations, matched, labels, mismatches, keys = {}, {}, {}, [], set()
    with samples_path.open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            cid, t = row['conversation_id'], str(int(row['timestep']))
            key = (cid, t)
            if key in keys:
                raise ValueError(f'Duplicate sample key in {samples_path}: {key}')
            keys.add(key)
            label = row['label']
            if label not in ('0', '1') or (cid in labels and labels[cid] != label):
                raise ValueError(f'Invalid/inconsistent label in {samples_path}: {cid}')
            labels[cid] = label
            conversations[cid] = conversations.get(cid, 0) + 1
            if key in lookup:
                matched[cid] = matched.get(cid, 0) + 1
                if 'utterances' in row:
                    texts = json.loads(row['utterances'])
                    if not isinstance(texts, list) or len(texts) != int(t):
                        raise ValueError(f'Invalid prefix text in {samples_path}: {key}')
                    if texts[-1].strip() != lookup[key]['text'].strip():
                        mismatches.append({'conversation_id': cid, 'timestep': t})
    complete = sorted(cid for cid in matched if matched[cid] == conversations[cid])
    return dict(samples=str(samples_path), total_conversations=len(conversations),
                matched_conversations=len(matched), complete_conversations=len(complete),
                complete_conversation_ids=complete,
                complete_by_label={label: sum(labels[c] == label for c in complete) for label in ('0', '1')},
                total_prefixes=len(keys), matched_prefixes=sum(matched.values()),
                text_mismatches=mismatches)


def enrich_samples(samples_path, annotations, output):
    """Left join last-observed-message annotations, keeping unmatched test rows."""
    lookup = {(r['conversation_id'], r['timestep']): r for r in annotations}
    extras = [c for c in FIELDS if c not in ('conversation_id', 'timestep', 'text')]
    column_map = {c: ('annotation_source_file' if c == 'source_file' else c) for c in extras}
    extras = list(column_map.values()) + ['annotation_matched', 'prefix_annotations']
    with samples_path.open(newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        if set(extras) & set(reader.fieldnames):
            raise ValueError('Input already contains annotation columns; use the original samples CSV')
        with output.open('w', newline='', encoding='utf-8') as target:
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames + extras)
            writer.writeheader()
            for row in reader:
                cid, timestep = row['conversation_id'], int(row['timestep'])
                annotation = lookup.get((cid, str(timestep)))
                row.update({dest: annotation.get(c, '') if annotation else '' for c, dest in column_map.items()})
                row['annotation_matched'] = str(annotation is not None).lower()
                # Missing messages stay null; never expose later/terminal attack annotations.
                row['prefix_annotations'] = json.dumps([
                    ({c: lookup[(cid, str(t))][c] for c in FIELDS if c != 'text'}
                     if (cid, str(t)) in lookup else None)
                    for t in range(1, timestep + 1)], ensure_ascii=False)
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, default=Path('data/pragmatic_annotations'))
    parser.add_argument('--source_dir', type=Path, help='Use local original Wiki CSVs without downloading')
    parser.add_argument('--revision', default=REVISION, help='Immutable Hugging Face commit SHA')
    parser.add_argument('--samples', type=Path, nargs='*', default=[] ,
                        help='Local prefix CSVs to audit, e.g. data/test_samplesLLMs.csv')
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        parser.error('--revision must be a full immutable 40-character commit SHA')
    if args.source_dir:
        files = sorted(str(path.resolve()) for path in args.source_dir.glob('*.csv'))
        revision = 'local'
    else:
        cache = args.output_dir / 'raw' / args.revision
        metadata = json.loads(download(f'https://huggingface.co/api/datasets/{REPO}/revision/{args.revision}',
                                       cache / 'metadata.json'))
        files = sorted(item['rfilename'] for item in metadata['siblings']
                       if item['rfilename'].startswith(SUBSET) and item['rfilename'].endswith('.csv'))
        base = f'https://huggingface.co/datasets/{REPO}/resolve/{args.revision}/'
        download(base + 'README.md', cache / 'README.md')
        revision = args.revision
    if not files:
        raise ValueError('No CGA-Wiki annotation CSVs found')
    annotations, sources, excluded = [], [], []
    for filename in files:
        content = (Path(filename).read_bytes() if args.source_dir else
                   download(base + quote(filename, safe='/'), cache / Path(filename).name))
        try:
            annotations.extend(normalize(content, filename, revision))
        except ValueError as exc:
            excluded.append(dict(file=filename, reason=str(exc)))
        sources.append(dict(file=filename, sha256=hashlib.sha256(content).hexdigest()))
    keys = [(row['conversation_id'], row['timestep']) for row in annotations]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate normalized annotation keys')
    audits = [coverage(path, annotations) for path in args.samples]
    mismatched_ids = {row['conversation_id'] for audit in audits for row in audit['text_mismatches']}
    annotations = [row for row in annotations if row['conversation_id'] not in mismatched_ids]
    usable_audits = [coverage(path, annotations) for path in args.samples]
    manifest = dict(dataset=f'https://huggingface.co/datasets/{REPO}', revision=revision,
                    source_dir=str(args.source_dir.resolve()) if args.source_dir else None,
                    subset=SUBSET, license='cc-by-nc-4.0', sources=sources,
                    conversations=len({r['conversation_id'] for r in annotations}),
                    messages=len(annotations), coverage=usable_audits, excluded_files=excluded,
                    alignment_before_exclusions=audits,
                    excluded_text_mismatch_ids=sorted(mismatched_ids),
                    notes=['Conversation IDs restored from filenames; original cell IDs retained.',
                           'timestep = zero-based source turn_index + 1.',
                           'Only annot1/annot2 and rater comments are available in this Wiki subset.',
                           'No inferred attacker, prompt type or Reddit annotations are added.',
                           'Full threads include future messages; analysis joins only observed prefixes.'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    output = args.output_dir / 'wiki_annotations.csv'
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(annotations)
    print(f'{output}: {manifest["conversations"]} conversations, {len(annotations)} messages; '
          f'{len(excluded)} invalid files and {len(mismatched_ids)} text-mismatched conversations '
          'excluded (see manifest.json)')
    for sample in args.samples:
        enrich_samples(sample, annotations, args.output_dir / (sample.stem + '_annotated.csv'))
    for audit in usable_audits:
        print(f"{audit['samples']}: {audit['complete_conversations']}/{audit['total_conversations']} "
              f"fully covered conversations; labels {audit['complete_by_label']}")


if __name__ == '__main__':
    main()
