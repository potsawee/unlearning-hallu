import json
import re
import sys, os


infile = sys.argv[1]

with open(infile) as fin:
    results = json.load(fin)

for name, result in results.items():
    hit = 0
    total = 0
    total_ent = 0
    total_top_p = 0
    for piece in result:
        answer = re.findall("[ABCD]", piece["pred"])
        answer = answer[0] if len(answer) >= 1 else ""
        if piece["ref"] == answer:
            hit += 1
        total_ent += piece["entropy"]
        total_top_p += piece["acc_prob"]
        total += 1
    acc = hit / total
    entropy = total_ent / total
    top_p = total_top_p / total
    print("{:.2f}".format(acc), '\t', "{:.2f}".format(entropy), '\t', "{:.2f}".format(top_p))