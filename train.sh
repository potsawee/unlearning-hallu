. /home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/MultiModal/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate videollama

export HF_HOME=/home/gs534/rds/hpc-work/work/ckpts/
expdir="exp/unlearning_bio_llama3_8B_selfcheck_50sample_schedule"
mkdir -p $expdir

python train.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 3 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --num_warmup_steps 0.0 \
    --weight_decay 0.0 \
    --lr_scheduler_type constant \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 50 \
    --save_interval 500 \
    --iterations 50000 \
    --train_data_path ./llm-geneation-prompts/mcqdata_full_gpt4o_train.json \
    --prompt_path ./llm-geneation-prompts/prompt.json \
    --lora_config config/lora_config.json \
    --selected_ids config/unlearn_ids.json \
    --resample_frequency 50 \
    --losstype mcqmemflatten \
    --npo_beta 0.005 \
    --retain_factor 0.5 \
    --selfchecksamples 50 \

# --load_from exp/unlearning_bio_llama3_8B_mcq_choices_moremem/checkpoint.6.final \
