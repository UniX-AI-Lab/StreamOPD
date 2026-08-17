#!/usr/bin/env python
"""
OVO-Bench scoring, using the official OVO-Bench implementation (inlined), plus the
B+R macro that the paper reports with and without the HLD sub-task.

Usage:
    python -m streamopd.scoring.ovo_bench --result_path results/ovo_run/ovo_results.json
"""

import argparse
import json
import os
import re


# ---------------------------------------------------------------------------
# OVOBenchScore (inlined from official OVO-Bench)
# ---------------------------------------------------------------------------

class OVOBenchOfflineScore:
    def __init__(self, args, results):
        self.args = args
        self.results = results

    def calculate_score_backward_realtime(self, results):
        def get_score(response, gt):
            if response is None:
                return 0
            return int(gt in response)
        for i in range(len(results)):
            results[i]["score"] = get_score(results[i]["response"], results[i]["ground_truth"])
        scores = {}
        for i in range(len(results)):
            if results[i]["task"] not in scores:
                scores[results[i]["task"]] = [results[i]["score"]]
            else:
                scores[results[i]["task"]].append(results[i]["score"])
        return results, scores

    def calculate_score_forward(self, results):
        def get_score_REC(response, gt):
            if response is None:
                return 0
            response = re.findall(r'\d+', response)
            response = "".join(response)
            return response == str(gt)

        def get_score_SSR_CRR(response, gt):
            if response is None:
                return 0
            return int(gt in response)

        scores = {}
        tasks = list(set([result["task"] for result in results]))
        for task in tasks:
            scores[task] = []
        for i, result in enumerate(results):
            if result["task"] == "REC":
                for j, test_info_ in enumerate(result["test_info"]):
                    scores["REC"].append(get_score_REC(test_info_["response"], test_info_["count"]))
            if result["task"] == "SSR":
                for j, test_info_ in enumerate(result["test_info"]):
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
                        scores["SSR"].append(1)
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    scores["SSR"].append(get_score_SSR_CRR(test_info_["response"], gt))
            if result["task"] == "CRR":
                for j, test_info_ in enumerate(result["test_info"]):
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
                        scores["CRR"].append(1)
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    scores["CRR"].append(get_score_SSR_CRR(test_info_["response"], gt))
        return results, scores

    def score(self):
        print(f"Offline Model: {self.args.model}")
        backward_results = self.results["backward"]
        realtime_results = self.results["realtime"]
        forward_results = self.results["forward"]
        per_task = {"backward": {}, "realtime": {}, "forward": {}}
        backward_score = realtime_score = forward_score = 0.0

        def report(label, title, results, scorer, bucket):
            print(title)
            _, scores = scorer(results)
            for k, v in scores.items():
                acc = sum(v) / len(v)
                per_task[bucket][k] = acc
                print(f"Task: {k}, Acc: {100 * acc:.2f}")
            avg = 100 * sum(per_task[bucket].values()) / len(per_task[bucket])
            print(f"{label} Avg.: {avg:.2f}\n")
            return avg

        if len(backward_results) > 0:
            backward_score = report("Backward", "Evaluate Backward Tracing...", backward_results,
                                    self.calculate_score_backward_realtime, "backward")
        if len(realtime_results) > 0:
            realtime_score = report("Realtime", "Evaluate Real-time Visual Perception...", realtime_results,
                                    self.calculate_score_backward_realtime, "realtime")
        if len(forward_results) > 0:
            forward_score = report("Forward", "Evaluate Forward Active Responding...", forward_results,
                                   self.calculate_score_forward, "forward")

        print(f"Total Avg.: {(backward_score + realtime_score + forward_score) / 3:.2f}")

        # B+R macro is the headline streaming metric: average the Backward and Realtime
        # category means. HLD (highlight detection) probes the ability to abstain, which
        # distillation degrades independently of streaming skill, so it is reported both
        # ways rather than silently folded in.
        if per_task["backward"] and per_task["realtime"]:
            print(f"B+R (with HLD): {(backward_score + realtime_score) / 2:.2f}")
            no_hld = {k: v for k, v in per_task["backward"].items() if k != "HLD"}
            if no_hld and len(no_hld) < len(per_task["backward"]):
                backward_no_hld = 100 * sum(no_hld.values()) / len(no_hld)
                print(f"B+R (without HLD): {(backward_no_hld + realtime_score) / 2:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate OVO-Bench evaluation scores')
    parser.add_argument("--model", type=str, default="model", help="label used in the printout and in --result_dir layout")
    parser.add_argument("--mode", type=str, default="offline", choices=["online", "offline"])
    parser.add_argument("--result_dir", type=str, default="results/ovo_bench", help="Directory containing results")
    parser.add_argument("--result_path", type=str, default=None, help="Direct path to a result JSON file (overrides --result_dir)")
    return parser.parse_args()


def load_results_from_dir(result_dir, model_name):
    model_dir = os.path.join(result_dir, model_name)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model results not found: {model_dir}")
    result_files = [f for f in os.listdir(model_dir) if f.endswith('.json')]
    if not result_files:
        raise FileNotFoundError(f"No result JSON files found in {model_dir}")
    results = {"backward": [], "realtime": [], "forward": []}
    for result_file in result_files:
        with open(os.path.join(model_dir, result_file), 'r') as f:
            result = json.load(f)
            results["backward"].extend(result.get("backward", []))
            results["realtime"].extend(result.get("realtime", []))
            results["forward"].extend(result.get("forward", []))
    return results


def load_results_from_path(result_path):
    with open(result_path, 'r') as f:
        result = json.load(f)
    return {
        "backward": result.get("backward", []),
        "realtime": result.get("realtime", []),
        "forward": result.get("forward", []),
    }


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"OVO-Bench Scoring")
    print(f"{'='*60}")

    if args.result_path:
        print(f"Result file: {args.result_path}")
        results = load_results_from_path(args.result_path)
    else:
        print(f"Model: {args.model}")
        print(f"Result Directory: {args.result_dir}")
        results = load_results_from_dir(args.result_dir, args.model)

    print(f"Backward: {len(results['backward'])}, Realtime: {len(results['realtime'])}, Forward: {len(results['forward'])}")
    print(f"{'='*60}\n")

    scorer = OVOBenchOfflineScore(args, results)
    scorer.score()

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
