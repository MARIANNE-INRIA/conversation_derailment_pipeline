"""CPU checks of threshold selection and the incomplete accumulation window."""
import os
os.environ.setdefault('USE_TF', '0')
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pandas as pd
import torch
from train_llm_forecasting import find_best_f1_threshold, train_one_epoch


class TrainingTests(unittest.TestCase):
    def test_threshold_search_aggregates_conversations(self):
        predictions = pd.DataFrame({'conversation_id':['a','a','b','b'],
                                    'label':[0,0,1,1], 'probability':[.1,.2,.1,.4]})
        tau, sweep = find_best_f1_threshold(predictions, .05, .95, .01)
        self.assertEqual(tau, .2)
        self.assertEqual(float(sweep['f1'].max()), 1.0)

    def test_last_window_has_full_gradient_and_scheduler_step(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[.1, -.1]]))
            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(logits=input_ids.float() @ self.weight)
        batch = {'input_ids':torch.ones(1,1), 'attention_mask':torch.ones(1,1),
                 'labels':torch.tensor([1])}
        accumulated = TinyModel()
        reference = TinyModel()
        optimizer = torch.optim.SGD(accumulated.parameters(), lr=.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        train_one_epoch(accumulated, [batch] * 5, optimizer, scheduler, torch.device('cpu'),
                        0, 1, 100, label_smoothing=0, gradient_accumulation_steps=2,
                        amp_dtype=torch.float32)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
        for _ in range(3):
            reference_optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(reference(**{k:v for k,v in batch.items() if k!='labels'}).logits, batch['labels'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            reference_optimizer.step()
        torch.testing.assert_close(accumulated.weight, reference.weight)
        self.assertEqual(scheduler.last_epoch, 3)


if __name__ == '__main__':
    unittest.main()
