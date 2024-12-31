. /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination

expdir="exp/unlearning_bio_llama3_1B_selfcheck_50sample_schedule"

step=1550
python inference.py \
    --model_path $expdir \
    --model_ckpt checkpoint.$step \
    --testfile /scratch/unlearning-hallu/llm-geneation-prompts/qa_testset_ext.json \
    --outfile $expdir/testoutput_$step.json \
    --logfile $expdir/testlog.txt \
    # --origmodel \