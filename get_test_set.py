import json
import sys, os


setid = sys.argv[1]
# sample_id = sys.argv[2]

with open("llm-geneation-prompts/WHPplus/whp_unlearn_testset_forget_obfuscat_more_yesno_all.json") as fin:
    all_questions = json.load(fin)

with open("llm-geneation-prompts/WHPplus/whp_unlearn_testset_forget_obfuscate_all.json") as fin:
    outside_questions = json.load(fin)

outputdir = "exp/unlearning_whp_llama3_8B_WHP_whp_{}_sample_20".format(setid)
with open(os.path.join(outputdir, "obfuscate_samples.json")) as fin:
    named_passages = json.load(fin)

in_probe_questions = {}
out_probe_questions = {}
gt_probe_questions = {}
for name, passages in named_passages.items():
    in_probe_questions[name] = []
    out_probe_questions[name] = []
    gt_probe_questions[name] = []
    for passage in passages:
        for question in all_questions[name][passage]:
            question["Question"] = "{} Directly answer Yes or No.".format(question["Question"])
        in_probe_questions[name].extend(all_questions[name][passage])
    for questions in outside_questions[name]:
        questions["Answer_questions"] = "{} Directly answer Yes or No.".format(questions["Answer_questions"])
        datapiece = {"Question": questions["Answer_questions"], "Answer": "Yes", "name": name}
        gt_probe_questions[name].append(datapiece)
        for question in questions["alternative_out_questions"]:
            question = "{} Only answer Yes or No.".format(question)
            datapiece = {"Question": question, "Answer": "No", "name": name}
            out_probe_questions[name].append(datapiece)

with open(os.path.join(outputdir, "in_probe_questions.json"), "w") as fout:
    json.dump(in_probe_questions, fout, indent=4)

with open(os.path.join(outputdir, "out_probe_questions.json"), "w") as fout:
    json.dump(out_probe_questions, fout, indent=4)

with open(os.path.join(outputdir, "gt_probe_questions.json"), "w") as fout:
    json.dump(gt_probe_questions, fout, indent=4)
