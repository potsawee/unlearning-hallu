import json
import random
from collections import Counter


# Define the file path
json_file_path = "new_whp_mcq_train.json"

# Load JSON data from the file
with open(json_file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Step 1: Fix misplaced "name" entries
def fix_json_structure(data):
    fixed_data = {}

    for key, entries in data.items():  # Iterate through all question IDs
        fixed_entries = []
        misplaced_name = None

        for entry in entries:
            if "question" in entry and "choices" in entry:
                # This is a valid question entry
                fixed_entries.append(entry)
            elif "name" in entry:
                # This is a misplaced name field
                misplaced_name = entry["name"]

        # If there was a misplaced "name", move it to the last valid entry
        if misplaced_name and fixed_entries:
            fixed_entries[-1]["name"] = misplaced_name  # Attach name to last valid question

        fixed_data[key] = fixed_entries  # Store fixed questions

    return fixed_data


# Step 2: Function to reorder choices and update correct answers
def process_and_fix_json(data):
    correct_order = ["A", "B", "C", "D", "E"]  # The correct order of choices

    for key in data:  # Iterate through all question IDs
        for entry in data[key]:  # Iterate through each question
            choices = entry["choices"]
            old_order_list = list(choices.keys())
            old_order_values = list(choices.values())

            old_answer = entry["answer"]

            if old_answer not in old_order_list:
                print(f"WARNING: Answer '{old_answer}' not found in choices for question: {entry['question']}")
                find_key = next((k for k, v in choices.items() if v == old_answer), None)
                if find_key is None:
                    old_answer = old_answer.split(":")[0].strip()
                    # import pdb; pdb.set_trace()
                else:
                    old_answer = find_key
                
            
            old_index = old_order_list.index(old_answer)
            new_answer = correct_order[old_index]
            new_choices = {correct_order[i]: old_order_values[i] for i in range(len(old_order_list))}

            # Update the entry
            entry["choices"] = new_choices
            entry["answer"] = new_answer  # Update correct answer key

    return data


fixed_structure_data = fix_json_structure(data)
fully_fixed_data = process_and_fix_json(fixed_structure_data)

# Step 3 Balance the answers
answer_counts = Counter()
for key in fully_fixed_data:
    for entry in fully_fixed_data[key]:
        answer_counts[entry["answer"]] += 1

# Calculate target per category (20% of total questions)
total_questions = sum(answer_counts.values())
target_per_answer = total_questions // 5  # 20% for each answer

print(f"Original Distribution: {answer_counts}")
print(f"Target per answer: {target_per_answer}")

#Identify overrepresented and underrepresented answersS
overrepresented = {k: v - target_per_answer for k, v in answer_counts.items() if v > target_per_answer}
underrepresented = {k: target_per_answer - v for k, v in answer_counts.items() if v < target_per_answer}

print(f"Overrepresented: {overrepresented}")
print(f"Underrepresented: {underrepresented}")

# Smart Swap Function (Ensures Logical Choices)
def smart_swap_answers(entry, from_answer, to_answer):
    """Swap choices while ensuring logical consistency."""
    choices = entry["choices"]

    # Ensure both keys exist in choices before swapping
    if from_answer in choices and to_answer in choices:
        # Swap answer positions
        choices[from_answer], choices[to_answer] = choices[to_answer], choices[from_answer]

        # Update the correct answer to new key
        entry["answer"] = to_answer
        return True  
    return False  

# Perform Smart Swaps
for key in fully_fixed_data:
    for entry in fully_fixed_data[key]:
        current_answer = entry["answer"]

        # If current answer is overrepresented
        if current_answer in overrepresented and overrepresented[current_answer] > 0:
            # Find an underrepresented answer
            possible_swaps = [k for k, v in underrepresented.items() if v > 0]
            
            if possible_swaps:
                new_answer = random.choice(possible_swaps)  # Pick a new answer

                # Swap only if the question still makes sense
                if smart_swap_answers(entry, current_answer, new_answer):
                    overrepresented[current_answer] -= 1
                    underrepresented[new_answer] -= 1

                    # Remove from dict if balanced
                    if overrepresented[current_answer] == 0:
                        del overrepresented[current_answer]
                    if underrepresented[new_answer] == 0:
                        del underrepresented[new_answer]

                # Stop when dataset is fully balanced
                if not overrepresented:
                    break
    if not overrepresented:
        break

# Verify New Distribution
new_answer_counts = Counter()
for key in fully_fixed_data:
    for entry in fully_fixed_data[key]:
        new_answer_counts[entry["answer"]] += 1

print(f"New Balanced Distribution: {new_answer_counts}")

# Save the balanced JSON file
balanced_json_file_path = "balanced_whp_mcq_train.json"
with open(balanced_json_file_path, "w", encoding="utf-8") as file:
    json.dump(fully_fixed_data, file, indent=4, ensure_ascii=False)

print(f"Balanced dataset has been saved to {balanced_json_file_path}")