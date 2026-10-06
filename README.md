# Conversation Derailment

This repository trains conversation-derailment classifiers, selects hyperparameters using validation data, and evaluates the selected model on a held-out test set. Training and evaluation jobs are submitted to a GPU cluster using Slurm.

## Install the environment

From the repository root, create and activate the Conda environment:

```bash
module load miniconda
conda env create -f environment.yml
conda activate PREDICHATE
```

For DeBERTa, use the encoder environment instead:

```bash
conda env create -f environment-encoders.yml
conda activate PREDICHATE_ENCODERS
```

Before submitting jobs, make sure the required Hugging Face models are downloaded and accessible from the cluster. Compute nodes may not have internet access. Check the environment with:

```bash
bash checks/check_environment.sh --models
```

## Prepare data

LLM configurations use the CSV data paths defined in their configuration files. RoBERTa and DeBERTa require tokenized data; prepare it before training:

```bash
sbatch jobs/prepare_encoder_data.sh
```

The HPO configurations are in `configs/hpo/`: Gemma 2, Gemma 3, Mistral, RoBERTa, and DeBERTa. Use the configuration matching your model.

## Train and select a model

Run the following from the repository root on the cluster. Create the log directory before submitting jobs:

```bash
mkdir -p logs
```

Replace `gemma3` below with the model/configuration you want to use. First run a baseline:

```bash
sbatch jobs/submit_hpo.sh \
  --config configs/hpo/gemma3.json \
  --output_dir hpo/gemma3 \
  --phase baseline
```

Then run hyperparameter search:

```bash
sbatch jobs/submit_hpo.sh \
  --config configs/hpo/gemma3.json \
  --output_dir hpo/gemma3 \
  --phase search --trials 20
```

After search completes, confirm the top configurations with multiple seeds:

```bash
sbatch jobs/submit_hpo.sh \
  --config configs/hpo/gemma3.json \
  --output_dir hpo/gemma3 \
  --phase confirm --top_k 3 --seeds 42 123 456
```

Training outputs, checkpoints, and the selected configuration are saved under the output directory. Do not run multiple HPO jobs against the same output directory at the same time.

## Evaluate on the test set

Run final evaluation only after confirmation has completed. The validation threshold is kept fixed; the test set is not used for model or threshold selection.

For LLMs, pass the test CSV:

```bash
sbatch jobs/evaluate_hpo.sh \
  --output_dir hpo/gemma3 \
  --test_path data/test_samplesLLMs.csv
```

For RoBERTa or DeBERTa, pass the prepared tokenized test file instead:

```bash
sbatch jobs/evaluate_hpo.sh \
  --output_dir hpo/roberta \
  --test_path encoder_data/roberta/test_samples.pt
```

The evaluation summary is written to `<output_dir>/test_summary.json`; per-run metrics and predictions are saved under the confirmation directory.

## Run tests

Run the repository's unit tests locally with:

```bash
python -m unittest discover -s tests -v
```

## Citation

If you use this pipeline in your research, please cite:

Fu, Yingxue, and Anaïs Ollagnier. 2026. [A Theoretically Grounded Approach to Summarizing Conversation Dynamics for Forecasting the Derailment of Online Conversations](https://aclanthology.org/2026.acl-long.243/). In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 5368–5384. Association for Computational Linguistics. DOI: [10.18653/v1/2026.acl-long.243](https://doi.org/10.18653/v1/2026.acl-long.243).

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff). Place this file at the repository root to enable GitHub’s “Cite this repository” feature.

```bibtex
@inproceedings{fu-ollagnier-2026-theoretically,
  title = {A Theoretically Grounded Approach to Summarizing Conversation Dynamics for Forecasting the Derailment of Online Conversations},
  author = {Fu, Yingxue and Ollagnier, Ana{\"i}s},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year = {2026},
  month = jul,
  publisher = {Association for Computational Linguistics},
  address = {San Diego, California, United States},
  pages = {5368--5384},
  doi = {10.18653/v1/2026.acl-long.243},
  url = {https://aclanthology.org/2026.acl-long.243/}
}
```
