from typing import *
import random
from Regex import *  # Regex, atomic_regexes_matching
from VSA import *

def precompute_heuristic(example: str, score_fn: Callable[[Regex, str], float]) -> Dict[int, float]:
    '''
    Calculates all values of the heuristic at the indices into the example
    '''
    scores = {len(example): 0.}  # At the end the score is 0.
    for i in reversed(range(len(example))):
        scores[i] = min(score_fn(regex, example[i:end]) + scores[end]
                for end in range(i+1, len(example)+1)
                for regex in atomic_regexes_matching(example[i:end]))
    return scores

class Heuristic(Protocol):
    def value_at(self, vsa_state: Tuple[int, ...]) -> float:
        ...

class MaxHeuristic:
    precomputed: List[Dict[int, float]]

    def __init__(self, examples: List[str]) -> float:
        self.precomputed = [
            precompute_heuristic(example, lambda regex, text: regex.simplicity_score() + regex.specificity_score(text))
            for example in examples
        ]

    def value_at(self, vsa_state: Tuple[int, ...]) -> float:
        assert len(vsa_state) == len(self.precomputed), 'Need the same number of indices as number of examples'
        return max(values[i] for values, i in zip(self.precomputed, vsa_state))

class TwoMaxHeuristic:
    '''Computes a bunch of pairwise stuff -- not all pairs, but many pairs'''
    permutation: List[int]
    precomputed: List[Dict[Tuple[int, int], float]]

    def __init__(self, examples: List[str]):
        self.permutation = list(range(len(examples)))
        random.shuffle(self.permutation)
        vsas = [VSA.single_example(ex) for ex in examples]
        self.precomputed = []
        for i in range(len(examples)):
            j0 = self.permutation[i]
            j1 = self.permutation[i-1]
            merged = vsas[j0].merge(vsas[j1])
            out = merged.get_best_regex(
                    lambda regex, text: regex.simplicity_score() +
                    sum(regex.specificity_score(t) for t in text))
            self.precomputed.append({k: v[0] for k, v in out})

    def value_at(self, vsa_state: Tuple[int, ...]) -> float:
        assert len(vsa_state) == len(self.precomputed), 'Need the same number of indices as number of examples'
        cur_max = 0.
        for i, data in enumerate(self.precomputed):
            j0 = self.permutation[i]
            j1 = self.permutation[i-1]
            score = data[vsa_state[j0], vsa_state[j1]]
            if score > cur_max:
                cur_max = score
        return cur_max

class SumHeuristic:
    precomputed: List[Dict[int, float]]
    
    def __init__(self, examples: List[str]):
        N = len(examples)
        self.precomputed = [
            precompute_heuristic(example, lambda regex, text: 1/N * regex.simplicity_score() + regex.specificity_score(text))
            for example in examples
        ]

    def value_at(self, vsa_state: Tuple[int, ...]) -> float:
        assert len(vsa_state) == len(self.precomputed), 'Need the same number of indices as number of examples'
        return sum(values[i] for values, i in zip(self.precomputed, vsa_state))

class AverageHeuristic:
    '''
    This is strictly worse than the max heuristic, so there's not much point in using it.
    But why not implement it.
    '''
    precomputed: List[Dict[int, float]]

    def __init__(self, examples: List[str]):
        self.precomputed = [
            precompute_heuristic(example, lambda regex, text:  regex.simplicity_score() + regex.specificity_score(text))
            for example in examples
        ]

    def value_at(self, vsa_state: Tuple[int, ...]) -> float:
        N = len(self.precomputed)
        assert len(vsa_state) == len(self.precomputed), 'Need the same number of indices as number of examples'
        return 1/N * sum(values[i] for values, i in zip(self.precomputed, vsa_state))

class TotalHeuristic:
    '''
    This is strictly worse than the max heuristic, so there's not much point in using it.
    But why not implement it.
    '''
    precomputed: List[Dict[int, float]]

    def __init__(self, examples: List[str]):
        self.precomputed = [
            precompute_heuristic(example, lambda regex, text:  regex.simplicity_score() + regex.specificity_score(text))
            for example in examples
        ]

    def value_at(self, vsa_state: Tuple[int, ...]) -> float:
        N = len(self.precomputed)
        assert len(vsa_state) == len(self.precomputed), 'Need the same number of indices as number of examples'
        return sum(values[i] for values, i in zip(self.precomputed, vsa_state))

class BestHeuristic:
    '''Computes both heuristics and takes the larger one'''
    max: MaxHeuristic
    sum: SumHeuristic

    def __init__(self, examples: List[str]):
        self.max = MaxHeuristic(examples)
        self.sum = SumHeuristic(examples)
    
    def value_at(self, vsa_state: Tuple[int, ...]) -> float:
        return max(self.max.value_at(vsa_state), self.sum.value_at(vsa_state))
