. /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination

expdir="exp/unlearning_bio_llama3_8B_mcq_choices_moremem_with_KL"
mkdir -p $expdir

python train.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 3 \
    --learning_rate 5e-4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --num_warmup_steps 0.0 \
    --weight_decay 0.0 \
    --lr_scheduler_type constant \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 50 \
    --save_interval 2500 \
    --iterations 50000 \
    --train_data_path ./llm-geneation-prompts/mcqdata_full_gpt4o_train.json \
    --prompt_path ./llm-geneation-prompts/prompt.json \
    --lora_config config/lora_config.json \
    --selected_ids config/unlearn_ids.json \
    --resample_frequency 50 \
    --losstype mcqmemflatten \
    --npo_beta 0.005 \
    --retain_factor 0.001 \
    --selfchecksamples 50 \

# --load_from exp/unlearning_bio_llama3_8B_mcq_choices_moremem/checkpoint.6.final \