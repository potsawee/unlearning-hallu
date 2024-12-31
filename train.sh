. /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination

expdir="exp/unlearning_bio_llama3_1B_selfcheck_50sample_schedule"
mkdir -p $expdir

python train.py \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --batch_size 1 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --num_warmup_steps 0.0 \
    --weight_decay 0.0 \
    --lr_scheduler_type constant \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 2 \
    --save_interval 50 \
    --iterations 50000 \
    --train_data_path /scratch/unlearning-hallu/llm-geneation-prompts/data-20241204.json \
    --prompt_path /scratch/unlearning-hallu/llm-geneation-prompts/prompt.json \
    --lora_config config/lora_config.json \
    --selected_id 10 \
    --resample_frequency 50 \
    --losstype selfcheck \
    --npo_beta 0.005 \
    --retain_factor 2.0 \
    --selfchecksamples 50 \