# Obfuscation vs. Unlearning
## Abstract
---
Unlearning has emerged as a critical capability for large language models (LLMs) to support data privacy, regulatory compliance, and ethical AI deployment. Recent techniques often rely on obfuscation by injecting incorrect or irrelevant information to suppress knowledge. Such methods effectively constitute knowledge addition rather than true removal, often leaving models vulnerable to probing. In this paper, we formally distinguish unlearning from obfuscation and introduce a probing-based evaluation framework to assess whether existing approaches genuinely remove targeted information. Moreover, we propose DF-MCQ, a novel unlearning method that flattens the model predictive distribution over automatically generated multiple-choice questions using KL-divergence, effectively removing knowledge about target individuals and triggering appropriate refusal behaviour. Experimental results demonstrate that DF-MCQ achieves unlearning with over 90\% refusal rate and a random choice-level uncertainty that is much higher than obfuscation on probing questions.


## Training Models with DF-MCQ
1. Dependencies
```
torch==2.1.0+cu118
transformers==4.47.1
tokenizers==0.21.0
```

2. Train
`bash train.sh`: Training script for DF-MCQ. Key parameters:
```
--selected_ids # Subset IDs, config/unlearn_ids1.json to config/unlearn_ids5.json
--retain_factor # The weight applied to the retain set loss
```

`bash train_whp.sh`: Training script for obfuscation. Key parameters:
```
--passage_id # Set to a specific ID or IDs separated by comma if you want to train with specific obfuscation passages. Set to -1 to use all passages.
--obfuscate_passages ./llm-geneation-prompts/WHPplus/all_obfuscate_samples.json # This file contains 20 passages for each individual.
```

3. Generate Training data
`python generate_mcq.py`: Note that you need to replace the name in the script. Example generated questions for training can be found at `./llm-geneation-prompts/WHPplus/balanced_whp_mcq_train_dedup.json`. This is processed with de-duplication and choice balancing (equal occurences of ABCDE).

## Probing Questions
### Yes-No Questions
- `llm-geneation-prompts/WHPplus/whp_unlearn_testset_forget_obfuscate_all.json`:\
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
    `alternative_in` are answers that are in the obfuscation passage (belong to one passage in `./llm-geneation-prompts/WHPplus/all_obfuscate_samples.json`).
    `alternative_out` are answers that are not in any of the obfuscation passages.
    `Answer_questions` is the question asking the right answer.

- `llm-geneation-prompts/WHPplus/whp_unlearn_testset_forget_obfuscat_more_yesno_all.json`:\
This file contains per-passage in-training set samples. 

- `llm-geneation-prompts/WHPplus/whp_unlearn_testset_retain_probe.json`:\
This file contains the retain set Yes-No probing questions.

- `llm-geneation-prompts/WHPplus/whp_unlearn_testset_hardretain_probe.json`:\
This file contains hard-retain set Yes-No probing questions

### MCQ Probing Questions
- `llm-geneation-prompts/WHPplus/whp_unlearn_testset_hardretain_mcq.json`:\
This file contains the hard-retrain set MCQ probing questions
- `llm-geneation-prompts/WHPplus/whp_unlearn_testset_obfuscate_mcq.json`:\
This file contains the forget set MCQ probing questions

## Evaluation
1. Run `bash eval.sh` to evaluate DF-MCQ. Set `testfile` to the right test file you want to run.
2. Run `bash eval_whp.sh` to evaluate WHP (obfuscation).
- __Note__: In order to run on the three sets of Yes-No probing questions, run `python get_test_set.py $subsetID` first where `$subsetID` is the subset of the forget set (1 to 5). This will generate the following 3 files under your experiment directory based on the obfuscation passages you use:
  - `in_probe_questions.json`
  - `out_probe_questions.json`
  - `gt_probe_questions.json`
Then run `eval_whp.sh` by correctly setting `testfile` to the correct file path.
