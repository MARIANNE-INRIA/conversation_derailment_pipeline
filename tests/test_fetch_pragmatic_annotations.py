import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fetch_pragmatic_annotations import normalize, coverage, enrich_samples


class AnnotationImportTests(unittest.TestCase):
    def source(self, indices=(0, 1)):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=['conversation_id', 'turn_index', 'text',
                                                   'annot1', 'annot2', 'speaker'])
        writer.writeheader()
        for i in indices:
            writer.writerow(dict(conversation_id='123456.0', turn_index=i, text=f'message {i}',
                                 annot1='5 + 19', annot2='quality', speaker='alice'))
        return stream.getvalue().encode()

    def test_restores_filename_id_and_shifts_zero_based_turns(self):
        rows = normalize(self.source(), '001.23.456_TOXIC_annotated.csv', 'revision')
        self.assertEqual(rows[0]['conversation_id'], '001.23.456')
        self.assertEqual([r['timestep'] for r in rows], ['1', '2'])
        self.assertEqual(rows[0]['source_conversation_id'], '123456.0')
        self.assertEqual(rows[0]['annot1'], '5 + 19')
        self.assertNotIn('attacker_id', rows[0])

    def test_enrichment_preserves_rows_and_never_includes_future(self):
        annotations = normalize(self.source(), '001_TOXIC_annotated.csv', 'revision')
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'test.csv', Path(tmp) / 'annotated.csv'
            source.write_text('conversation_id,timestep,label,source_file\n001,1,1,original.csv\n002,1,0,control.csv\n')
            enrich_samples(source, annotations, output)
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['source_file'], 'original.csv')
            self.assertEqual(rows[0]['annotation_matched'], 'true')
            prefix = json.loads(rows[0]['prefix_annotations'])
            self.assertEqual(len(prefix), 1)
            self.assertEqual(prefix[0]['timestep'], '1')
            self.assertEqual(rows[1]['annotation_matched'], 'false')
            self.assertEqual(rows[1]['annot1'], '')
            self.assertEqual(json.loads(rows[1]['prefix_annotations']), [None])

    def test_rejects_ambiguous_positions(self):
        for indices in [(0, 0), (0, 2), (1, 2)]:
            with self.assertRaises(ValueError):
                normalize(self.source(indices), '001_TOXIC_annotated.csv', 'revision')

    def test_coverage_checks_text_and_preserves_negative_denominator(self):
        annotations = normalize(self.source(), '001_TOXIC_annotated.csv', 'revision')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'samples.csv'
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['conversation_id', 'timestep', 'label', 'utterances'])
                writer.writeheader()
                for cid, label, text in [('001', '1', 'message 0'), ('002', '0', 'other')]:
                    writer.writerow(dict(conversation_id=cid, timestep=1, label=label,
                                         utterances=json.dumps([text])))
            audit = coverage(path, annotations)
            self.assertEqual(audit['total_conversations'], 2)
            self.assertEqual(audit['complete_by_label'], {'0': 0, '1': 1})
            self.assertEqual(audit['text_mismatches'], [])
            annotations[0]['text'] = 'wrong message'
            self.assertEqual(len(coverage(path, annotations)['text_mismatches']), 1)


if __name__ == '__main__':
    unittest.main()
