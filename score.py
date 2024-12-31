import json
import re
import sys, os


infile = sys.argv[1]

with open(infile) as fin:
    results = json.load(fin)

for name, result in results.items():
    hit = 0
    total = 0
    for piece in result:
        answer = re.findall("[ABCD]", piece["pred"])
        answer = answer[0] if len(answer) >= 1 else ""
        if piece["ref"] == answer:
            hit += 1
        total += 1
    # print("{}: {}/{} = {}".format(name, hit, total, hit/total))
    print(hit)