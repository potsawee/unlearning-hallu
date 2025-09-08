mode="mcqmembothflatten"

# expdir="exp/unlearning_whp_llama2_7Bfull_MCQ_${mode}_1"
modelpath=Qwen/Qwen2.5-7B-Instruct
traindata=./data/WHPplus/balanced_whp_mcq_train_dedup.json
expdir="exp/unlearning_whp_qwen25_7B_MCQ_${mode}_5_mem1.0"
mkdir -p $expdir

python scripts/train.py \
    --model_path $modelpath \
    --batch_size 8 \
    --learning_rate 30e-5 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --num_warmup_steps 0.0 \
    --weight_decay 0.0 \
    --lr_scheduler_type linear \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 50 \
    --save_interval 1000 \
    --iterations 50000 \
    --train_data_path $traindata \
    --prompt_path ./data/prompt.json \
    --lora_config ./config/lora_config.json \
    --selected_ids ./config/unlearn_ids5.json \
    --resample_frequency 50 \
    --losstype $mode \
    --npo_beta 0.05 \
    --retain_factor 0.1 \
    --selfchecksamples 20 \
