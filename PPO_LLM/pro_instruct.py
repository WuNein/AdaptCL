import json

# Load the file
new_data = []

with open("/root/data/ppo4cl/fliter/test_gen_neg_raw.jsonl", 'r') as file:
    for line in file:
        json_obj = json.loads(line.strip())
        new_data.append(json_obj)

# Process each dictionary in the list to remove "</s>" and quotation marks at the start and end of the sentence
for i, d in enumerate(new_data):
    hard_value = d["hard"]

    # Remove "</s>" if it exists
    hard_value = hard_value.replace("</s>", "").strip()

    # Check if the sentence ends with a quotation mark and remove it
    if hard_value.endswith('"'):
        hard_value = hard_value[:-1]

    # Check if the sentence starts with a quotation mark and remove it
    if hard_value.startswith('"'):
        hard_value = hard_value[1:]

    # Update the "hard" value in the dictionary
    new_data[i]["hard"] = hard_value

# Save the processed data to a new file
with open("/root/data/ppo4cl/fliter/test_gen_neg_raw.json", 'w') as file:
    json.dump(new_data, file)
