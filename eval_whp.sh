. /home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/MultiModal/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate videollama

epoch=1
step=final
# setname=hardretain
# setname=forget
setid=$1
# passage_id=$2
nsample=20
expdir="exp/unlearning_whp_llama3_8B_WHP_whp_${setid}_sample_${nsample}"
# expdir="exp/unlearning_whp_llama3_8B_WHP_whp_${setid}_${passage_id}_sample_${nsample}"
# expdir="exp/unlearning_whp_llama3_8Bfull_MCQ_mcqmembothflatten_${setid}_mem1.0"

# python inference.py \
#     --model_path $expdir \
#     --model_ckpt checkpoint.$epoch.$step \
#     --testfile llm-geneation-prompts/WHPplus/mcq_to_yesno_probe.json \
#     --outfile $expdir/obfuscate_probe_answers.json \
#     --nsamples 1 \
#     --logfile $expdir/testlog.txt \
#     --origmodel \
# 
# echo Finished gt probe

python inference.py \
    --model_path $expdir \
    --model_ckpt checkpoint.$epoch.$step \
    --testfile $expdir/gt_probe_questions.json \
    --outfile $expdir/gt_probe_answers.json \
    --nsamples 1 \
    --logfile $expdir/testlog.txt \
    --origmodel \

python inference.py \
    --model_path $expdir \
    --model_ckpt checkpoint.$epoch.$step \
    --testfile $expdir/in_probe_questions.json \
    --outfile $expdir/in_probe_answers.json \
    --nsamples 1 \
    --logfile $expdir/testlog.txt \
    --origmodel \

echo Finished in probe

python inference.py \
    --model_path $expdir \
    --model_ckpt checkpoint.$epoch.$step \
    --testfile $expdir/out_probe_questions.json \
    --outfile $expdir/out_probe_answers.json \
    --nsamples 1 \
    --logfile $expdir/testlog.txt \
    --origmodel \

echo Finished out probe

# python inference.py \
#     --model_path $expdir \
#     --model_ckpt checkpoint.$epoch.$step \
#     --testfile llm-geneation-prompts/WHPplus/whp_unlearn_testset_hardretain_probe.json \
#     --outfile $expdir/hardretain_probe_testoutput_${epoch}_${step}.json \
#     --nsamples 1 \
#     --logfile $expdir/testlog.txt \
#     # --origmodel \
# 
# echo Finished hard retain
# 
# python inference.py \
#     --model_path $expdir \
#     --model_ckpt checkpoint.$epoch.$step \
#     --testfile llm-geneation-prompts/WHPplus/whp_unlearn_testset_retain_probe.json \
#     --outfile $expdir/retain_probe_testoutput_${epoch}_${step}.json \
#     --nsamples 1 \
#     --logfile $expdir/testlog.txt \
#     # --origmodel \
# 
# echo Finished ALL
