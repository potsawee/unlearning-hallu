# Unlearning vs. Obfuscation

- This is the official implementation of the paper [Unlearning vs. Obfuscation: Are We Truly Removing Knowledge?](https://arxiv.org/abs/2505.02884).
- The paper is to appear at EMNLP 2025 Main Conference.

## Abstract
---
Unlearning has emerged as a critical capability for large language models (LLMs) to support data privacy, regulatory compliance, and ethical AI deployment. Recent techniques often rely on obfuscation by injecting incorrect or irrelevant information to suppress knowledge. Such methods effectively constitute knowledge addition rather than true removal, often leaving models vulnerable to probing. In this paper, we formally distinguish unlearning from obfuscation and introduce a probing-based evaluation framework to assess whether existing approaches genuinely remove targeted information. Moreover, we propose **DF-MCQ**, a novel unlearning method that flattens the model predictive distribution over automatically generated multiple-choice questions using KL-divergence, effectively removing knowledge about target individuals and triggering appropriate refusal behaviour. Experimental results demonstrate that DF-MCQ achieves unlearning with over 90\% refusal rate and a random choice-level uncertainty that is much higher than obfuscation on probing questions.


## Training Models with DF-MCQ

### 1. Dependencies
We use the following dependencies:
```
torch==2.1.0+cu118
transformers==4.47.1
tokenizers==0.21.0
```

### 2. Usage Instructions

All commands should be run from the repository root directory. The scripts are organized in the `scripts/` directory, but they expect to be run from the root directory so they can properly import from `models.py` and `dataloader.py`, and access files in `data/` and `config/` directories.

```bash
# Make sure you're in the project root
cd /path/to/unlearning-hallu
```

### 3. Training

This repository implements these unlearning approaches:

#### 3.1) DF-MCQ Method (Our ProposedApproach)
Flattens the model predictive distribution over multiple-choice questions to achieve true knowledge removal.

```bash
# Basic training with default settings
bash scripts/train.sh

# For other model variants
bash scripts/train_qwen.sh  # For Qwen model
```

Key parameters you can modify in the script:
- `--selected_ids`: Choose which subset to forget (config/unlearn_ids1.json to unlearn_ids5.json)  
- `--retain_factor`: Weight for retain set loss (default: 1.0)

#### 3.2) WHP Obfuscation Method (Baseline Comparison)
Uses obfuscation by injecting incorrect information rather than true unlearning.

```bash
bash scripts/train_whp.sh
# set passage_id to -1 for all passages
```

Training script for obfuscation. Key parameters:
- `--passage_id`: Set to a specific ID or IDs separated by comma if you want to train with specific obfuscation passages. Set to -1 to use all passages.
- `--obfuscate_passages`: This file contains 20 passages for each individual.

### 4. Data Generation
The training data is already provided, but you can generate new MCQ data:

```bash
python scripts/generate_mcq.py
```

Note that you need to replace the name in the script. Example generated questions for training can be found at `./data/WHPplus/balanced_whp_mcq_train_dedup.json`. This is processed with de-duplication and choice balancing (equal occurences of ABCDE).

## Probing Questions
### Yes-No Questions
- `data/WHPplus/whp_unlearn_testset_forget_obfuscate_all.json`:\
This file contains a dictionary of all 10 people in the forget set. Each person has a list of questions where each question has the follow structure:
    ```
    {
        "Question": "What nationality was P. A. Yeomans?",
        "Answer": "Australian",
        "name": "P. A. Yeomans",
        "alternative_in": [
            "American",
            ...
        ],
        "alternative_out": [
            "British",
            ...
        ],
        "Answer_questions": "Was P. A. Yeomans Australian?",
        "alternative_in_questions": [
            "Was P. A. Yeomans American?",
            ...
        ],
        "alternative_out_questions": [
            "Was P. A. Yeomans British?",
    `         ...
        ]
    }
    ```
    `alternative_in` are answers that are in the obfuscation passage (belong to one passage in `./data/WHPplus/all_obfuscate_samples.json`).
    `alternative_out` are answers that are not in any of the obfuscation passages.
    `Answer_questions` is the question asking the right answer.

- `data/WHPplus/whp_unlearn_testset_forget_obfuscat_more_yesno_all.json`:\
This file contains per-passage in-training set samples. 

- `data/WHPplus/whp_unlearn_testset_retain_probe.json`:\
This file contains the retain set Yes-No probing questions.

- `data/WHPplus/whp_unlearn_testset_hardretain_probe.json`:\
This file contains hard-retain set Yes-No probing questions

### MCQ Probing Questions
- `data/WHPplus/whp_unlearn_testset_hardretain_mcq.json`:\
This file contains the hard-retrain set MCQ probing questions
- `data/WHPplus/whp_unlearn_testset_obfuscate_mcq.json`:\
This file contains the forget set MCQ probing questions

## Evaluation
1. Run `bash scripts/eval.sh` to evaluate DF-MCQ. Set `testfile` to the right test file you want to run.
2. Run `bash scripts/eval_whp.sh` to evaluate WHP (obfuscation).
- __Note__: In order to run on the three sets of Yes-No probing questions, run `python scripts/get_test_set.py $subsetID` first where `$subsetID` is the subset of the forget set (1 to 5). This will generate the following 3 files under your experiment directory based on the obfuscation passages you use:
  - `in_probe_questions.json`
  - `out_probe_questions.json`
  - `gt_probe_questions.json`
Then run `scripts/eval_whp.sh` by correctly setting `testfile` to the correct file path.

## Citation
```
@article{sun2025unlearning,
  title={Unlearning vs. Obfuscation: Are We Truly Removing Knowledge?},
  author={Sun, Guangzhi and Manakul, Potsawee and Zhan, Xiao and Gales, Mark},
  journal={arXiv preprint arXiv:2505.02884},
  year={2025}
}
```

## Contact
If you have any questions, please contact Guangzhi Sun (gs534@cam.ac.uk), Potsawee Manakul (potsawee@scb10x.com).