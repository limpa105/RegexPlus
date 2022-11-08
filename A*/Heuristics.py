from typing import *
from Regex import *  # Regex, atomic_regexes_matching

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
