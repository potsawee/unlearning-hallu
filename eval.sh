. /home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/MultiModal/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate videollama


export CUDA_VISIBLE_DEVICES=2
# expdir="exp/unlearning_whp_llama3_8B_WHP_whp_1_sample_20"
setid=$1
# expdir="exp/unlearning_whp_llama3_8Bfull_MCQ_mcqmembothflatten_${setid}_mem1.0"
expdir="exp/unlearning_whp_llama3_8B_WHP_whp_${setid}_sample_20"
# expdir="exp/unlearning_whp_qwen25_7B_MCQ_mcqmembothflatten_5_mem1.0"

epoch=1
step=final
setname=hardretain_mcq
# setname=obfuscate_mcq
# setname=hardretain

python inference.py \
    --model_path $expdir \
    --model_ckpt checkpoint.$epoch.$step \
    --testfile llm-geneation-prompts/WHPplus/whp_unlearn_testset_${setname}.json \
    --outfile $expdir/${setname}_testoutput_${epoch}_${step}.json \
    --logfile $expdir/testlog.txt \
    # --origmodel \
    # --nsamples 101 \
    # --do_selfcheck \

setname=obfuscate_mcq
# python inference.py \
#     --model_path $expdir \
#     --model_ckpt checkpoint.$epoch.$step \
#     --testfile llm-geneation-prompts/WHPplus/whp_unlearn_testset_${setname}.json \
#     --outfile $expdir/${setname}_testoutput_${epoch}_${step}.json \
#     --logfile $expdir/testlog.txt \
#     # --origmodel \
#     # --nsamples 101 \
#     # --do_selfcheck \
