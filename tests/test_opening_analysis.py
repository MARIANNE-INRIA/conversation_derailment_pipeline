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
from scipy.stats import binomtest, fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from opening_analysis import (add_opening_arguments, analyze_openings, compute_opening_results,
                              effect, feature_set, opening_cohort, significance_marks)
from error_analysis import run


def arguments(output):
    parser = argparse.ArgumentParser()
    add_opening_arguments(parser)
    args = parser.parse_args([])
    args.output_dir = str(output)
    args.sep = ','
    args.min_support = 1
    args.multilabel_separator = r'[+;|]'
    args.opening_min_count = 1
    args.opening_features = ['politeness_strategy', 'prompt_type']
    return args


def data():
    rows = []
    # B, C, attacker not present, same author, unknown, two controls.
    for cid,label,authors,attacker in [('b',1,['a','z'],'a'), ('c',1,['a','z'],'z'),
            ('late',1,['a','z'],'other'), ('same',1,['a','a'],'a'),
            ('unknown',1,['a','z'],''), ('n1',0,['a','z'],''), ('n2',0,['a','z'],'')]:
        for pos in [1,2]:
            rows.append(dict(conversation_id=cid, timestep=pos, label=label, author=authors[pos-1],
                             attacker_id=attacker, politeness_strategy='direct' if label else 'greeting',
                             prompt_type='factual' if pos == 1 else 'coordination', _merge='both'))
    return pd.DataFrame(rows)


class OpeningTests(unittest.TestCase):
    def test_effect_orientation_smoothing_and_both_tests(self):
        row = effect(8,10,2,10,.5)
        self.assertAlmostEqual(row['log_odds'],np.log((8.5*8.5)/(2.5*2.5)))
        self.assertAlmostEqual(row['binomial_p'],binomtest(8,10,.2).pvalue)
        self.assertAlmostEqual(row['fisher_p'],fisher_exact([[8,2],[2,8]])[1])
        self.assertLess(row['ci_low'],row['log_odds'])
        self.assertGreater(row['ci_high'],row['log_odds'])
        self.assertAlmostEqual(effect(2,10,8,10)['log_odds'],-row['log_odds'])
        self.assertTrue(np.isfinite(effect(0,10,4,10)['log_odds']))
        self.assertTrue(np.isnan(effect(0,0,4,10)['log_odds']))

    def test_missing_is_not_feature_absence(self):
        self.assertIsNone(feature_set('', True, '[+]'))
        self.assertEqual(feature_set('false', True, '[+]'),frozenset())
        self.assertEqual(feature_set('[]', False, '[+]'),frozenset())
        self.assertEqual(feature_set('a+b', False, '[+]'),frozenset(['a','b']))
        with self.assertRaises(ValueError):
            feature_set('unknown',True,'[+]')

    def test_significance_conventions(self):
        for p,expected in [(.05,''),(.049,'*'),(.009,'**'),(.0009,'***')]:
            self.assertEqual(significance_marks(p,'*'),expected)
            self.assertEqual(significance_marks(p,'+'),expected.replace('*','+'))

    def test_roles_and_same_position_controls(self):
        args = arguments('/tmp/unused')
        first, second, cohort = opening_cohort(data(),args)
        self.assertEqual(cohort.loc['b','role_group'],'attacker_initiated')
        self.assertEqual(cohort.loc['c','role_group'],'non_attacker_initiated')
        self.assertEqual(cohort.loc['late','role_group'],'attacker_not_in_opening_exchange')
        self.assertEqual(cohort.loc['same','role_group'],'same_author_in_both_openings')
        table, _, _ = compute_opening_results(first,second,cohort,args)
        b = table.loc[(table.panel == 'B') & (table.feature == 'politeness_strategy') & (table.category == 'direct')]
        self.assertTrue((b.n_awry == 1).all())
        self.assertTrue((b.n_on_track == 2).all())
        self.assertEqual(b.set_index('position').role.to_dict(),{1:'attacker',2:'non_attacker'})
        c = table.loc[(table.panel == 'C') & (table.feature == 'politeness_strategy') & (table.category == 'direct')]
        self.assertEqual(c.set_index('position').role.to_dict(),{1:'non_attacker',2:'attacker'})

    def test_complete_openings_feature_missingness_and_no_attacker_guess(self):
        frame=data()
        frame.loc[(frame.conversation_id == 'b') & (frame.timestep == 2),'_merge']='left_only'
        frame.loc[(frame.conversation_id == 'c') & (frame.timestep == 1),'politeness_strategy']=''
        frame=frame.drop(columns=['author','attacker_id'])
        frame['aggressive']='OAG'
        args=arguments('/tmp/unused')
        first,second,cohort=opening_cohort(frame,args)
        self.assertNotIn('b',cohort.index)
        self.assertTrue((cohort.loc[cohort.label == 1,'role_group']=='missing_attacker_or_author').all())
        table,coverage,_=compute_opening_results(first,second,cohort,args)
        self.assertFalse(table.loc[table.panel.isin(['B','C']),'supported'].any())
        self.assertEqual(coverage[0]['missing_feature_conversations'],1)

    def test_metadata_override_preserves_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'meta.csv'
            path.write_text('conversation_id,attacker_id\nc,a\n')
            args=arguments(tmp)
            args.opening_metadata=str(path)
            _,_,cohort=opening_cohort(data(),args)
            self.assertEqual(cohort.loc['c','role_group'],'attacker_initiated')
            self.assertEqual(cohort.loc['b','role_group'],'missing_attacker_or_author')

    def test_integration_without_predictions_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            frame=data()
            frame[['conversation_id','timestep','label']].to_csv(root/'pred.csv',index=False)
            frame.drop(columns=['label','_merge']).to_csv(root/'ann.csv',index=False)
            args=arguments(root/'report')
            args.opening_only=True
            args.predictions=str(root/'pred.csv')
            args.annotations=str(root/'ann.csv')
            args.threshold=None
            args.threshold_json=None
            args.annotation_conversation_col='conversation_id'
            args.annotation_turn_col='timestep'
            args.filename_ids=False
            with contextlib.redirect_stdout(io.StringIO()):
                audit=run(args)
            self.assertEqual(audit['eligible_conversations'],7)
            for name in ['report.md','audit.json','cohort.csv','log_odds.csv',
                         'opening_log_odds.png','opening_log_odds.svg','opening_log_odds.pdf']:
                self.assertGreater((root/'report/opening_analysis'/name).stat().st_size,0)
            table=pd.read_csv(root/'report/opening_analysis/log_odds.csv')
            self.assertEqual(set(table.panel),{'A','B','C'})
            np.testing.assert_array_equal(table.solid,table.significant & (table.log_odds.abs() >= .2))


if __name__ == '__main__':
    unittest.main()
