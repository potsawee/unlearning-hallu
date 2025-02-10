. /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination

expdir="exp/unlearning_whp_llama3_8B_MCQ_mcqmemflattenA_1"

epoch=2
step=final
setname=hardretain
# setname=forget

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
    --testfile llm-geneation-prompts/WHPplus/whp_unlearn_testset_${setname}.json \
    --outfile $expdir/${setname}_testoutput_${epoch}_${step}.json \
    --logfile $expdir/testlog.txt \
    # --origmodel \