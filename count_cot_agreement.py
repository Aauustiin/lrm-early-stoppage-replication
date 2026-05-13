import json
import re

with open("gsm-sft-aug-results.json") as f:
    data = json.load(f)

total = 0
matches = 0

for entry in data:
    reasoning = entry["model_reasoning"]
    answers = entry["model_answers"]

    # answers[i] was produced after i reasoning steps, so for i >= 1,
    # the most recent CoT step is reasoning[i-1]
    for i, step in enumerate(reasoning):
        answer = answers[i + 1]
        # steps are formatted as <<expression=result>>
        m = re.search(r"=([^=>\s]+)>>", step)
        if m is None:
            continue
        cot_result = m.group(1).strip()
        total += 1
        if answer == cot_result:
            matches += 1

print(f"Matches: {matches} / {total} = {matches / total:.4f}")
