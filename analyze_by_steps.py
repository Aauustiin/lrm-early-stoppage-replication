import json
import argparse
from collections import defaultdict
from datasets import load_dataset


def count_gold_steps(answer: str) -> int:
    """Count reasoning steps in a GSM8k gold answer (lines before ####)."""
    lines = [l.strip() for l in answer.strip().split("\n") if l.strip()]
    return sum(1 for l in lines if not l.startswith("####"))


def main(results_path: str):
    print(f"Loading GSM8k test split...", flush=True)
    dataset = load_dataset("gsm8k", "main", split="test")

    gold_steps = [count_gold_steps(ex["answer"]) for ex in dataset]

    print(f"Loading results from {results_path}...", flush=True)
    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]
    assert len(results) == len(dataset), (
        f"Result count {len(results)} != dataset size {len(dataset)}"
    )

    buckets = defaultdict(
        lambda: defaultdict(lambda: {"first_match_frac": [], "stable_match_frac": [], "num_steps": []})
    )

    for entry in results:
        idx = entry["sample_idx"]
        num_gold_steps = gold_steps[idx]
        is_correct = entry["original_result"]["is_correct"][-1]
        first_match_frac = entry["original_result"]["first_match_frac"]
        stable_match_frac = entry["original_result"]["stable_match_frac"]
        num_steps = entry["original_result"]["num_steps"]

        buckets[num_gold_steps][is_correct]["first_match_frac"].append(first_match_frac)
        buckets[num_gold_steps][is_correct]["stable_match_frac"].append(stable_match_frac)
        buckets[num_gold_steps][is_correct]["num_steps"].append(num_steps)

    output = {}

    for num_steps in sorted(buckets):
        output[num_steps] = {}
        for is_correct in buckets[num_steps]:
            fm = buckets[num_steps][is_correct]["first_match_frac"]
            sm = buckets[num_steps][is_correct]["stable_match_frac"]
            steps_count = buckets[num_steps][is_correct]["num_steps"]
            output[num_steps][is_correct] = {
                "average_first_match_frac": sum(fm) / len(fm),
                "average_stable_match_frac": sum(sm) / len(sm),
                "count": len(fm),
                "average_num_steps": sum(steps_count) / len(steps_count) if steps_count else 0,
            }

    out_path = "sft_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_file",
        nargs="?",
        default="results/coconut.json",
        help="Path to a results JSON file (default: results/coconut.json)",
    )
    args = parser.parse_args()
    main(args.results_file)