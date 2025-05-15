mode="whp"
nsample=20
setid=$1
passage_id=-1
expdir="exp/unlearning_whp_llama3_8B_WHP_${mode}_${setid}_sample_${nsample}"
# expdir="exp/unlearning_whp_llama2_7B_MCQ_${mode}_1"
mkdir -p $expdir
modelname=meta-llama/Llama-3.1-8B-Instruct

python train_whp.py \
    --model_path $modelname \
    --batch_size 1 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --num_warmup_steps 0.05 \
    --weight_decay 0.0 \
    --lr_scheduler_type constant \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 50 \
    --save_interval 20000 \
    --iterations 50000 \
    --train_data_path ./llm-geneation-prompts/WHPplus/whp_names.json \
    --prompt_path ./llm-geneation-prompts/prompt.json \
    --lora_config config/lora_config.json \
    --selected_ids config/unlearn_ids${setid}.json \
    --resample_frequency 50 \
    --losstype $mode \
    --npo_beta 0.005 \
    --retain_factor 0.0 \
    --selfchecksamples $nsample \
    --passage_id $passage_id \
    --obfuscate_passages ./llm-geneation-prompts/WHPplus/all_obfuscate_samples.json \
    # --obfuscate_passages exp/unlearning_whp_llama3_8B_WHP_whp_${setid}_sample_20/obfuscate_samples.json \
    # --passage_id $passage_id \
