. /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination


expdir="exp/unlearning_bio_llama3_NPO"
mkdir -p $expdir

python train.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 1 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --num_warmup_steps 0.0 \
    --weight_decay 0.0 \
    --lr_scheduler_type cosine \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 1 \
    --save_interval 200 \
    --iterations 20000 \
    --train_data_path /scratch/unlearning-hallu/llm-geneation-prompts/data-20241204.json \
    --prompt_path /scratch/unlearning-hallu/llm-geneation-prompts/prompt.json \
    --lora_config config/lora_config.json \
    --selected_id 10 \
    --resample_frequency 20 \
    --losstype selfcheckdpo \
    --npo_beta 1.0 \
    --retain_factor 1.0 \
