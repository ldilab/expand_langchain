import json
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel


class Evaluator(BaseModel):
    path: Path
    gt_key: str
    filter_keys: List[str] = []
    filter_weights: List[float] = []

    data: List[dict] = None

    def __init__(self, **data):
        super().__init__(**data)

        with open(self.path, "r") as f:
            self.data = json.load(f)

    def _load_results(self, key):
        l = []
        for datum in self.data:
            passed = datum[key]
            if isinstance(passed, list):
                if isinstance(passed[0], bool):
                    l.append([[int(p)] for p in passed])
                elif isinstance(passed[0], list):
                    if len(passed[0]) == 0:
                        l.append([0 for p in passed])
                    else:
                        l.append([np.mean(p) for p in passed])
                else:
                    raise ValueError("Invalid format")
            else:
                l.append([[passed]])

        return np.array(l)

    def _run_fixed_k(
        self,
        k,
        n,
    ):
        metrics = {}

        gt_results = self._load_results(self.gt_key)  # shape: (164,10,1)
        gt_results = gt_results.reshape(len(gt_results), -1)
        naive_pass_rate = pass_at_k(gt_results, k)
        metrics["naive"] = naive_pass_rate

        total_prob_count = len(gt_results)
        pass_rates = {}
        for filter_key in self.filter_keys:
            filter_results = self._load_results(filter_key)
            pass_rates[filter_key] = filter_results

        scores = []
        for prob_idx in range(total_prob_count):
            score = 0
            for key, weight in zip(self.filter_keys, self.filter_weights):
                score += weight * pass_rates[key][prob_idx]

            scores.append(score)

        metrics["weighted"] = pass_at_k(gt_results, k, scores)

        return metrics

    def run(
        self,
        k: List[int],
        n: int,
    ):
        results = {}
        for _k in k:
            results[f"pass@{_k}"] = self._run_fixed_k(_k, n)

        return results


def probability(n, c, k):
    return 1 - comb(n - c, k) / comb(n, k)


def naive_pass_at_k(
    gt_results: List[List[int]],
    k: int,
):
    num_correct = [sum(x) for x in gt_results]
    num_samples = [len(x) for x in gt_results]

    probs = np.array([probability(n, c, k) for n, c in zip(num_samples, num_correct)])
    return probs.mean()


def pass_at_k_with_scores_per_problem(gt_results, pass_at_k, scores):
    score_mapping = defaultdict(int)
    score_count = defaultdict(int)
    for score, gt_result in zip(scores, gt_results):
        score = int(score)
        score_mapping[score] += gt_result
        score_count[score] += 1

    score_sorted = sorted(score_count.keys(), reverse=True)
    cumsum = 0
    correct_so_far = 0

    for score in score_sorted:
        cumsum += score_count[score]
        if cumsum > pass_at_k:
            if correct_so_far:
                return 1
            else:
                n = score_count[score]
                k = score_mapping[score]
                x = pass_at_k + score_count[score] - cumsum
                if k == n:
                    return 1
                else:
                    return 1 - comb(n - k, x) / comb(n, x)
        if score_mapping[score] > 0:
            correct_so_far = 1
        if cumsum == pass_at_k:
            return correct_so_far


def pass_at_k_with_scores(
    gt_results: List[List[int]],
    k: int,
    scores: List[List[float]],
):
    prob_results = []
    for _scores, _gt_results in zip(scores, gt_results):
        prob_results.append(pass_at_k_with_scores_per_problem(_gt_results, k, _scores))

    return np.mean(prob_results)


def pass_at_k(
    gt_results: List[List[int]],
    k: int,
    scores: Optional[List[List[float]]] = None,
):
    if scores is None:
        return naive_pass_at_k(gt_results, k)
    else:
        return pass_at_k_with_scores(gt_results, k, scores)
