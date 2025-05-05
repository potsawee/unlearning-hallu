import json
import math
import re
import sys, os


infile = sys.argv[1]

with open("llm-geneation-prompts/WHPplus/whp_names.json") as fin:
    data = json.load(fin)
forget_set = [p["name"] for p in data if "passage" in p]

forget_set_spec = forget_set
if len(sys.argv) > 1:
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
falsein_prob = 0
total = 0
retain_hit = 0
total_retain = 0
total_entropy = 0

for name, result in results.items():
    for piece in result:
        if name in forget_set and name in forget_set_spec:
            if piece["pred"] == piece["ref"]:
                hit += 1
            falsein_prob += piece["Choice_distribution"][piece["False_in"]] if "False_in" in piece else 0
            total_entropy += piece["entropy"]
            total += 1
forget_acc = hit / max(1, total)
print("Forget set Accuracy")
print("{:.2f}".format(forget_acc*100))
print("Forget set entropy")
print("{:.2f}".format(total_entropy / max(1, total)))
print("Forget set False in prob")
print("{:.2f}".format(falsein_prob / max(1, total)))
