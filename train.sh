. /home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/MultiModal/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate videollama

export HF_HOME=/home/gs534/rds/hpc-work/work/ckpts/

mode="mcqmem"

expdir="exp/unlearning_whp_llama3_8B_MCQ_${mode}_1"
mkdir -p $expdir

python train.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 3 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --num_warmup_steps 0.0 \
    --weight_decay 0.0 \
    --lr_scheduler_type constant \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 50 \
    --save_interval 20000 \
    --iterations 50000 \
    --train_data_path ./llm-geneation-prompts/WHPplus/balanced_whp_mcq_train_dedup.json \
    --prompt_path ./llm-geneation-prompts/prompt.json \
    --lora_config config/lora_config.json \
    --selected_ids config/unlearn_ids.json \
    --resample_frequency 50 \
    --losstype $mode \
    --npo_beta 0.005 \
    --retain_factor 0.05 \
    --selfchecksamples 50 \

# expdir="exp/unlearning_whp_llama3_8B_MCQ_${mode}_2"
# mkdir -p $expdir
# 
# python train.py \
#     --model_path meta-llama/Llama-3.1-8B-Instruct \
#     --batch_size 3 \
#     --learning_rate 5e-5 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 10 \
#     --num_warmup_steps 0.0 \
#     --weight_decay 0.0 \
#     --lr_scheduler_type constant \
#     --outputdir $expdir \
#     --logfile $expdir/log.txt \
#     --log_interval 50 \
#     --save_interval 20000 \
#     --iterations 50000 \
#     --train_data_path ./llm-geneation-prompts/WHPplus/new_whp_mcq_train.json \
#     --prompt_path ./llm-geneation-prompts/prompt.json \
#     --lora_config config/lora_config.json \
#     --selected_ids config/unlearn_ids2.json \
#     --resample_frequency 50 \
#     --losstype $mode \
#     --npo_beta 0.005 \
#     --retain_factor 0.2 \
#     --selfchecksamples 50 \
# 
# expdir="exp/unlearning_whp_llama3_8B_MCQ_${mode}_3"
# mkdir -p $expdir
# 
# python train.py \
#     --model_path meta-llama/Llama-3.1-8B-Instruct \
#     --batch_size 3 \
#     --learning_rate 5e-5 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 10 \
#     --num_warmup_steps 0.0 \
#     --weight_decay 0.0 \
#     --lr_scheduler_type constant \
#     --outputdir $expdir \
#     --logfile $expdir/log.txt \
#     --log_interval 50 \
#     --save_interval 20000 \
#     --iterations 50000 \
#     --train_data_path ./llm-geneation-prompts/WHPplus/new_whp_mcq_train.json \
#     --prompt_path ./llm-geneation-prompts/prompt.json \
#     --lora_config config/lora_config.json \
#     --selected_ids config/unlearn_ids3.json \
#     --resample_frequency 50 \
#     --losstype mcqmem \
#     --npo_beta 0.005 \
#     --retain_factor 0.2 \
#     --selfchecksamples 50 \
# 
# expdir="exp/unlearning_whp_llama3_8B_MCQ_sample_${mode}_4"
# mkdir -p $expdir
# 
# python train.py \
#     --model_path meta-llama/Llama-3.1-8B-Instruct \
#     --batch_size 3 \
#     --learning_rate 5e-5 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 10 \
#     --num_warmup_steps 0.0 \
#     --weight_decay 0.0 \
#     --lr_scheduler_type constant \
#     --outputdir $expdir \
#     --logfile $expdir/log.txt \
#     --log_interval 50 \
#     --save_interval 20000 \
#     --iterations 50000 \
#     --train_data_path ./llm-geneation-prompts/WHPplus/whp_mcq_train.json \
#     --prompt_path ./llm-geneation-prompts/prompt.json \
#     --lora_config config/lora_config.json \
#     --selected_ids config/unlearn_ids4.json \
#     --resample_frequency 50 \
#     --losstype mcqmem \
#     --npo_beta 0.005 \
#     --retain_factor 0.2 \
#     --selfchecksamples 50 \
# 
# expdir="exp/unlearning_whp_llama3_8B_MCQ_sample_${mode}_5"
# mkdir -p $expdir
# 
# python train.py \
#     --model_path meta-llama/Llama-3.1-8B-Instruct \
#     --batch_size 3 \
#     --learning_rate 5e-5 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 10 \
#     --num_warmup_steps 0.0 \
#     --weight_decay 0.0 \
#     --lr_scheduler_type constant \
#     --outputdir $expdir \
#     --logfile $expdir/log.txt \
#     --log_interval 50 \
#     --save_interval 20000 \
#     --iterations 50000 \
#     --train_data_path ./llm-geneation-prompts/WHPplus/whp_mcq_train.json \
#     --prompt_path ./llm-geneation-prompts/prompt.json \
#     --lora_config config/lora_config.json \
#     --selected_ids config/unlearn_ids5.json \
#     --resample_frequency 50 \
#     --losstype mcqmem \
#     --npo_beta 0.005 \
#     --retain_factor 0.2 \
#     --selfchecksamples 50 \
# 
# # --load_from exp/unlearning_bio_llama3_8B_mcq_choices_moremem/checkpoint.6.final \
