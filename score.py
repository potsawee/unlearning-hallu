import json
import re
import sys, os
from rouge_score import rouge_scorer


infile = sys.argv[1]
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

with open("llm-geneation-prompts/WHPplus/whp_names.json") as fin:
    data = json.load(fin)
forget_set = [p["name"] for p in data if "passage" in p]

with open(infile) as fin:
    results = json.load(fin)

for name, result in results.items():
    hit = 0
    total = 0
    total_ent = 0
    total_top_p = 0
    retain_hit = 0
    total_retain = 0
    for piece in result:
        if len(piece["ref"]) == 1:
            answer = re.findall("[ABCD]", piece["pred"])
            answer = answer[0] if len(answer) >= 1 else ""
            if piece["ref"] == answer:
                hit += 1
        else:
            scores = scorer.score(piece["ref"], piece["pred"])
            hit += scores["rougeL"].recall
        total_ent += piece["entropy"]
        total_top_p += piece["acc_prob"]
        total += 1
    acc = hit / total
    entropy = total_ent / total
    top_p = total_top_p / total
    print(name)
    print("{:.2f}".format(acc), '\t', "{:.2f}".format(entropy), '\t', "{:.2f}".format(top_p))
