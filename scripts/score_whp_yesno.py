import json
import math
import re
import sys, os


infile = sys.argv[1]

with open("../data/WHPplus/whp_names.json") as fin:
    data = json.load(fin)
forget_set = [p["name"] for p in data if "passage" in p]

forget_set_spec = []
if len(sys.argv) > 2:
    setid = sys.argv[2]
    if setid == "all":
        forget_set_spec = ["Benedetto Varchi", "Wilhelm Wattenbach", "Dany Robin", "Martin Gutzwiller", "P. A. Yeomans", "Karl Hartl", "Michaela Dorfmeister", "Leo Slezak", "Alicia de Larrocha", "Christian Krohg"]
    else:
        forget_ids = {
            "1": ["Benedetto Varchi", "Wilhelm Wattenbach"],
            "2": ["Dany Robin", "Martin Gutzwiller"],
            "3": ["P. A. Yeomans", "Karl Hartl"],
            "4": ["Michaela Dorfmeister", "Leo Slezak"],
            "5": ["Alicia de Larrocha", "Christian Krohg"]
        }
        forget_set_spec = forget_ids[setid]
        print("Particularly for {}".format(forget_set_spec))

one_passage_pairs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
two_passage_pairs = ["0,1", "2,3", "4,5", "6,7", "8,9", "10,11", "12,13", "14,15", "16,17", "18,19"]
results = {}
if "all" in infile:
    print("="*89)
    print(infile)
    print("="*89)
    for i in one_passage_pairs:
        if "allin" in infile:
            with open("exp/unlearning_whp_llama3_8B_WHP_whp_{}_{}_sample_1/in_probe_answers.json".format(setid, i)) as fin:
                result = json.load(fin)
        elif "allout" in infile:
            with open("exp/unlearning_whp_llama3_8B_WHP_whp_{}_{}_sample_1/out_probe_answers.json".format(setid, i)) as fin:
                result = json.load(fin)
        elif "allgt" in infile:
            with open("exp/unlearning_whp_llama3_8B_WHP_whp_{}_{}_sample_1/gt_probe_answers.json".format(setid, i)) as fin:
                result = json.load(fin)
        else:
            with open("exp/unlearning_whp_llama3_8B_WHP_whp_{}_{}_sample_1/hardretain_probe_answers.json".format(setid, i)) as fin:
                result = json.load(fin)
        for name, answers in result.items():
            if name not in results:
                results[name] = []
            results[name].extend(answers)
else:
    with open(infile) as fin:
        results = json.load(fin)

TP = 0
FP = 0
FN = 0
hit = 0
total = 0
retain_hit = 0
total_retain = 0
total_entropy = 0

for name, result in results.items():
    for piece in result:
        if (name in forget_set and name in forget_set_spec) or forget_set_spec == []:
            # if piece["ref"] == "Yes":
            #     continue
            if isinstance(piece["pred"][0], str):
                yes = 0
                for pred in piece["pred"]:
                    if "Yes" in pred:
                        yes += 1
                yes_prob = yes / len(piece["pred"])
            else:
                yes_prob = piece["pred"][0]
            # print("Yes" if yes_prob > 0.5 else "No")
            if "retain" in infile:
                piece["ref"] = "No"
            no_prob = 1 - yes_prob
            if yes_prob >= no_prob and piece["ref"] == "Yes":
                TP += 1
                hit += 1
            elif yes_prob < no_prob and piece["ref"] == "Yes":
                FN += 1
            elif yes_prob >= no_prob and piece["ref"] == "No":
                FP += 1
            else:
                hit += 1
            yes_prob = max(0.01, min(yes_prob, 0.99))
            entropy = yes_prob * math.log(yes_prob) + (1 - yes_prob) * math.log(1 - yes_prob)
            total_entropy -= entropy
            total += 1
forget_acc = hit / max(1, total)
retain_acc = retain_hit / max(1, total_retain)
print("TP FP FN")
print(TP, FP, FN)
print("Forget set Acc")
print("{:.2f}".format(forget_acc*100))
# print("Forget set Precision")
# print("{:.2f}".format(TP/(TP+FP)*100))
# print("Forget set Recall")
# print("{:.2f}".format(TP/(TP+FN)*100))
print("Forget set entropy")
print("{:.2f}".format(total_entropy / total))
print("Retain set names")
print("{:.2f}".format(retain_acc*100))
