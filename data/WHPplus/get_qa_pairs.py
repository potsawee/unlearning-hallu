import json


with open("balanced_whp_mcq_train_dedup.json") as fin:
    data = json.load(fin)

questions = []
for nameid, qlist in data.items():
    
