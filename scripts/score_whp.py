import json
import re
import sys, os
from rouge_score import rouge_scorer


infile = sys.argv[1]
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

with open("../data/WHPplus/whp_names.json") as fin:
    data = json.load(fin)
forget_set = [p["name"] for p in data if "passage" in p]

forget_set_spec = forget_set
if len(sys.argv) > 2:
    setid = sys.argv[2]
    forget_ids = {
        "1": ["Benedetto Varchi", "Wilhelm Wattenbach"],
        "2": ["Dany Robin", "Martin Gutzwiller"],
        "3": ["P. A. Yeomans", "Karl Hartl"],
        "4": ["Michaela Dorfmeister", "Leo Slezak"],
        "5": ["Alicia de Larrocha", "Christian Krohg"]
    }
    forget_set_spec = forget_ids[setid]
    print("Particularly for {}".format(forget_set_spec))

with open(infile) as fin:
    results = json.load(fin)

hit = 0
total = 0
retain_hit = 0
total_retain = 0

for name, result in results.items():
    for piece in result:
        scores = scorer.score(piece["ref"], piece["pred"])
        if name in forget_set and name in forget_set_spec:
            hit += scores["rougeL"].recall
            print(scores["rougeL"].recall)
            total += 1
        elif name not in forget_set:
            retain_hit += scores["rougeL"].recall
            total_retain += 1
forget_acc = hit / max(1, total)
retain_acc = retain_hit / max(1, total_retain)
print("Forget set names")
print("{:.2f}".format(forget_acc*100))
print("Retain set names")
print("{:.2f}".format(retain_acc*100))
