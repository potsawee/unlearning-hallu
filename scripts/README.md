# Scripts Directory

This directory contains all executable scripts for training, evaluation, and data generation. All scripts should be run from the **repository root directory**.

## Training Scripts

### DF-MCQ Method (Our Approach)
- **`train.py`** - Main Python training script for DF-MCQ method
- **`train.sh`** - Bash script for training Llama models with DF-MCQ
- **`train_qwen.sh`** - Bash script for training Qwen models with DF-MCQ

### WHP Obfuscation Method (Baseline)
- **`train_whp.py`** - Python training script for WHP obfuscation method
- **`train_whp.sh`** - Bash script for WHP obfuscation training

## Evaluation Scripts

- **`inference.py`** - Model inference and evaluation script
- **`eval.sh`** - Evaluation script for DF-MCQ models
- **`eval_whp.sh`** - Evaluation script for WHP models

## Scoring Scripts

- **`score.py`** - General scoring for model evaluation
- **`score_whp.py`** - Scoring for WHP method evaluation
- **`score_whp_mcq.py`** - MCQ-specific scoring for WHP method
- **`score_whp_yesno.py`** - Yes/No question scoring for WHP method

## Data Generation Scripts

- **`generate_mcq.py`** - Generate multiple-choice questions for training
- **`generate_questions.py`** - Generate probing questions for evaluation
- **`get_test_set.py`** - Prepare test sets for specific evaluation subsets

## Utility Scripts

- **`load.py`** - Model loading utilities

## Usage Examples

```bash
# Training with DF-MCQ (run from repository root)
bash scripts/train.sh

# Training with WHP obfuscation
bash scripts/train_whp.sh

# Evaluation
bash scripts/eval.sh

# Generate MCQ data
python scripts/generate_mcq.py

# Prepare test sets for subset 1
python scripts/get_test_set.py 1
```

## Notes

1. **Always run from repository root**: All scripts expect to be executed from the main repository directory, not from within the `scripts/` folder.

2. **Configuration**: Most scripts use configuration files from `../config/` and data from `../data/` (relative to the script location).

3. **Output**: Training scripts will create experiment directories in `../exp/` with model checkpoints and logs.

4. **Dependencies**: Ensure you have installed the required dependencies listed in the main README before running any scripts.
