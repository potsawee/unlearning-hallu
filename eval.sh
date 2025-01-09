. /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination

expdir="exp/unlearning_bio_llama3_8B_mcq_choices_moremem_with_entropy_k1"

epoch=2
step=final

# python inference.py \
#     --model_path $expdir \
#     --model_ckpt checkpoint.$step \
#     --testfile /scratch/unlearning-hallu/llm-geneation-prompts/qa_testset_ext.json \
#     --outfile $expdir/testoutput_${step}.json \
#     --logfile $expdir/testlog.txt \
#     # --origmodel \

python inference.py \
    --model_path $expdir \
    --model_ckpt checkpoint.$epoch.$step \
    --testfile /scratch/unlearning-hallu/llm-geneation-prompts/qa_testset_ext.json \
    --outfile $expdir/testoutput_${epoch}_${step}.json \
    --logfile $expdir/testlog.txt \
    # --origmodel \