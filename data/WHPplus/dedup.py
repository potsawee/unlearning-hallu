import json


with open("balanced_whp_mcq_train.json") as fin:
    data = json.load(fin)

dedup_data = {}
for person, questions in data.items():
    dedup_data[person] = []
    dup_count = 0
    covered_question = set()
    for question in questions:
        q = question["question"].lower()
        if q not in covered_question and "which" not in q:
            covered_question.add(q)
            dedup_data[person].append(question)
        else:
            dup_count += 1
    print(dup_count)

with open("balanced_whp_mcq_train_dedup.json", "w") as fout:
    json.dump(dedup_data, fout, indent=4)
