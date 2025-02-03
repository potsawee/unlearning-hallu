. /home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/MultiModal/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate videollama

expdir="exp/unlearning_whp_llama3_8B_MCQ_mcqmem_1"

epoch=3
step=final

# python inference.py \
#     --model_path $expdir \
#     --model_ckpt checkpoint.$step \
#     --testfile /scratch/unlearning-hallu/llm-geneation-prompts/qa_testset_ext.json \
#     --outfile $expdir/testoutput_${step}.json \
#     --logfile $expdir/testlog.txt \
#     # --origmodel \

# python inference.py \
#     --model_path $expdir \
#     --model_ckpt checkpoint.$epoch.$step \
#     --testfile llm-geneation-prompts/WHPplus/whp_unlearn_testset_retain.json \
#     --outfile $expdir/retain_testoutput_${epoch}_${step}.json \
#     --logfile $expdir/testlog.txt \
#     # --origmodel \

python inference.py \
    --model_path $expdir \
    --model_ckpt checkpoint.$epoch.$step \
    --testfile llm-geneation-prompts/WHPplus/whp_unlearn_testset_forget.json \
    --outfile $expdir/forget_testoutput_${epoch}_${step}.json \
    --logfile $expdir/testlog.txt \
    # --origmodel \
