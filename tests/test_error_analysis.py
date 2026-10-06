import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from error_analysis import (associations, bh_adjust, boolean_tokens, build_units, load_annotations,
                            parse_inferences, prepare_predictions, run, tokens)


class ErrorAnalysisTests(unittest.TestCase):
    def test_codes_and_booleans(self):
        self.assertEqual(tokens('22 + 14 + 22'), frozenset(['22','14']))
        self.assertEqual(tokens('PRE/IMP'), frozenset(['pre/imp']))
        self.assertEqual(boolean_tokens('False'), frozenset(['false']))
        self.assertEqual(boolean_tokens('yes'), frozenset(['true']))
        self.assertEqual(boolean_tokens('uncertain'), frozenset(['unknown:uncertain']))

    def test_text_is_not_assigned_an_invented_count(self):
        self.assertEqual(parse_inferences('[{"content":"hello"},{"content":"bye"}]'), ('structured',2.0))
        self.assertEqual(parse_inferences('{"content":"hello"}, {"content":"bye"}'), ('structured',2.0))
        self.assertEqual(parse_inferences('literal'), ('none_or_literal',0.0))
        self.assertTrue(np.isnan(parse_inferences('some free text')[1]))

    def test_bh_known_values(self):
        np.testing.assert_allclose(bh_adjust([.01,.04,.03,np.nan]), [.03,.04,.04,np.nan], equal_nan=True)

    def test_binary_association_and_correct_fp_denominator(self):
        units = pd.DataFrame({'label':[0]*8+[1]*2, 'error':[1]*4+[0]*4+[1]*2})
        values = pd.Series([frozenset(['cag'])]*4+[frozenset(['nag'])]*4+[frozenset(['cag'])]*2)
        cats, _ = associations(units, {'aggressive':values}, {}, 1, True)
        row = cats.loc[(cats.category=='cag') & (cats.target=='false_positive')].iloc[0]
        self.assertEqual(row.n_with,4)
        self.assertEqual(row.n_without,4)
        self.assertEqual(row.phi,1)
        self.assertAlmostEqual(row.p_value,2/70)
        self.assertEqual(row.risk_difference,1)
        descriptive, _ = associations(units, {'aggressive':values}, {}, 1, False)
        self.assertTrue(descriptive.p_value.isna().all())

    def test_conversation_aggregation_and_rater_agreement(self):
        joined = pd.DataFrame({'conversation_id':['a','a','b'], 'timestep':[1,2,1],
                               'label':[1,1,0], 'prediction':[1,0,0],
                               'annot1':['22+14','22','1'], 'annot2':['14+22','35','1']})
        units, cats, _ = build_units(joined, 'conversation', r'[+;|]')
        self.assertEqual(len(units),2)
        self.assertEqual(units.loc['a','error_type'],'TP')
        self.assertEqual(cats['annotator_agreement']['a'],frozenset(['agree','disagree']))

    def test_duplicate_predictions_and_future_turns_rejected(self):
        frame = pd.DataFrame({'conversation_id':['a'], 'timestep':['1'], 'total_turns':['2'],
                              'label':['1'],'probability':['.5']})
        self.assertEqual(prepare_predictions(frame,.5).prediction.iloc[0],0)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            prepare_predictions(pd.concat([frame,frame]),.5)
        frame['timestep']='2'
        with self.assertRaisesRegex(ValueError,'terminal'):
            prepare_predictions(frame,.5)

    def test_annotation_conflicts_and_embedded_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'ann.csv'
            pd.DataFrame({'conversation_id':['001','001'], 'timestep':['1','1'],
                          'annot1':['22','35']}).to_csv(file,index=False)
            with self.assertRaisesRegex(ValueError,'Conflicting'):
                load_annotations(file,'conversation_id','timestep',False,',')
            pd.DataFrame({'conversation_id':['001'], 'timestep':['1'],
                          'message_info':[json.dumps([{'annot1':'22','annot2':'35'}])]}).to_csv(file,index=False)
            ann = load_annotations(file,'conversation_id','timestep',False,',')
            self.assertEqual(ann.conversation_id.iloc[0],'001')
            self.assertEqual(ann.annot2.iloc[0],'35')

    def test_full_report_excludes_future_labels_and_checks_missing_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pd.DataFrame({'conversation_id':['001','001','002'], 'timestep':[1,2,1],
                          'total_turns':[3,3,1], 'label':[1,1,0], 'probability':[.1,.2,.9]}).to_csv(root/'pred.csv',index=False)
            pd.DataFrame({'conversation_id':['001','001','001','002'], 'timestep':[1,2,3,1],
                          'aggressive':['NAG','CAG','OAG','NAG'], 'annot1':['22','14','99','1'],
                          'annot2':['22','35','99','1'], 'as_intended':['yes','no','yes','true'],
                          'Pragmatic_Inferences':['[]','literal','future','[]']}).to_csv(root/'ann.csv',index=False)
            args=argparse.Namespace(predictions=str(root/'pred.csv'),annotations=str(root/'ann.csv'),
                output_dir=str(root/'out'),threshold=.5,threshold_json=None,level='conversation',
                annotation_conversation_col='conversation_id',annotation_turn_col='timestep',
                filename_ids=False,sep=',',min_support=1,multilabel_separator=r'[+;|]',allow_partial=False)
            with contextlib.redirect_stdout(io.StringIO()):
                audit=run(args)
            self.assertEqual(audit['confusion'],dict(TP=0,TN=0,FP=1,FN=1))
            cats=pd.read_csv(root/'out/categorical_associations.csv')
            self.assertNotIn('oag',set(cats.category))
            self.assertTrue((root/'out/report.md').exists())
            ann=pd.read_csv(root/'ann.csv',dtype=str).iloc[1:]
            ann.to_csv(root/'ann.csv',index=False)
            with self.assertRaisesRegex(ValueError,'no annotation'):
                run(args)
            args.allow_partial=True
            with contextlib.redirect_stdout(io.StringIO()):
                audit=run(args)
            self.assertEqual(audit['excluded_conversations'],1)
            self.assertEqual(audit['analyzed_units'],1)


if __name__ == '__main__':
    unittest.main()
