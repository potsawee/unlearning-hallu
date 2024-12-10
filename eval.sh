. /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination

expdir="exp/unlearning_bio_llama3_rewrite"

python inference.py \
    --model_path $expdir \
    --model_ckpt checkpoint.2000 \
    --testfile /scratch/unlearning-hallu/llm-geneation-prompts/qa_testset.json \
    --outfile $expdir/testoutput_2000.json \
    --logfile $expdir/testlog.txt \
    # --origmodel \