import json

with open("mcq_to_yesno.json") as fin:
    data = json.load(fin)

for name, questions in data.items():
    for datapiece in questions:
        datapiece["Question"] += " Only answer Yes or No."

with open("mcq_to_yesno_probe.json", "w") as fout:
    json.dump(data, fout, indent=4)
