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
    data: List[Tuple[Tuple[int, ...], Dict[Tuple[int, ...], float]]]

    def __init__(self, examples: List[str]):
        if len(examples) == 0:
            self.data = []
            return
        score_fn = lambda regex, texts: regex.simplicity_score() \
                + sum(regex.specificity_score(t) for t in texts)
        if len(examples) == 1:
            vsa = VSA.single_example(examples[0])
            self.data = [(
                (0,),
                {k: v[0] for k, v in vsa.all_best_regexes(score_fn).items()})]
            return
        permutation = list(range(len(examples)))
        random.shuffle(permutation)
        vsas = [VSA.single_example(ex) for ex in examples]
        self.data = []
        for i in range(len(examples)):
            j0 = permutation[i]
            j1 = permutation[i-1]
            both = vsas[j0].merge(vsas[j1])
            self.data.append((
                (j0, j1),
                {k: v[0] for k, v in both.all_best_regexes(score_fn).items()}))

    def value_at(self, vsa_state: Tuple[int, ...]) -> float:
        return max(data[tuple(vsa_state[i] for i in ids)] for ids, data in self.data)

class TwoSumHeuristic:
    '''Computes a bunch of pairwise stuff -- not all pairs, but many pairs'''
    data: List[Tuple[Tuple[int, ...], Dict[Tuple[int, ...], float]]]

    def __init__(self, examples: List[str]):
        N = len(examples)
        if len(examples) <= 1:
            raise Exception('nope')
        score_fn = lambda regex, texts: 1/N*regex.simplicity_score() \
                + sum(regex.specificity_score(t) for t in texts)
        permutation = list(range(len(examples)))
        random.shuffle(permutation)
        vsas = [VSA.single_example(ex) for ex in examples]
        self.data = []
        for i in range(len(examples)):
            j0 = permutation[i]
            j1 = permutation[i-1]
            both = vsas[j0].merge(vsas[j1])
            self.data.append((
                (j0, j1),
                {k: v[0] for k, v in both.all_best_regexes(score_fn).items()}))

    def value_at(self, vsa_state: Tuple[int, ...]) -> float:
        return sum(data[tuple(vsa_state[i] for i in ids)] for ids, data in self.data)

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
    An inadmissable heuristic
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

class NoHeuristic:
    '''Zero all the time. Guaranteed admissible. For testing the search code.'''
    def __init__(self, examples: List[str]):
        pass
    def value_at(self, vsa_state: Tuple[int, ...]) -> float:
        return 0.
