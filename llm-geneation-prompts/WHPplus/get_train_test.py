import json


with open("whp_mcq_train.json") as fin:
    traindata = json.load(fin)

with open("../mcqdata_full_gpt4o_train.json") as fin:
    othertrain = json.load(fin)

with open("whp_names.json") as fin:
    people = json.load(fin)

forget_set = [str(p["id"]) for p in people if "passage" in p]
newdata = {}
for idx in forget_set:
    newdata[idx] = traindata[idx]

for idx, samples in othertrain.items():
    newdata[idx] = samples[:500]

with open("new_whp_mcq_train.json", "w") as fout:
    json.dump(newdata, fout, indent=4)

# with open("whp_unlearn_testset.json") as fin:
#     testdata = json.load(fin)
# 
# forget_set = [str(p["name"]) for p in people if "passage" in p]
# retain_set = {}
# forget_only_set = {}
# for name, samples in testdata.items():
#     if name not in forget_set:
#         retain_set[name] = samples
#     else:
#         forget_only_set[name] = samples
# 
# with open("whp_unlearn_hard_retain.json") as fin:
#     harddata = json.load(fin)
# 
# for name, samples in harddata.items():
#     if name not in retain_set:
#         retain_set[name] = samples
#     else:
#         print(name)
# 
# with open("whp_unlearn_testset_all.json", "w") as fout:
#     json.dump(retain_set, fout, indent=4)
# 
# with open("whp_unlearn_testset_forget.json", "w") as fout:
#     json.dump(forget_only_set, fout, indent=4)
