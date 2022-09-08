import next_chars
import numpy as np
from typing import *

def poisson(k):
    # Unnormalized Poisson distribution with mean 4
    return 4**k / np.math.factorial(k)

class Regex(next_chars.DFA):
    def __init__(self, components: List[str], MAX_LENGTH=50,
            LENGTH_DISTRIB=poisson):
        super().__init__(next_chars.regex_to_nfa(components))
        self.MAX_LENGTH = MAX_LENGTH

        # mat is float[lengths × nodes]
        mat = np.zeros((MAX_LENGTH+1, len(self.nodes)))
        for node_id, node in enumerate(self.nodes):
            if node.is_end:
                mat[0,node_id] = 1
        for prev_len in range(MAX_LENGTH):
            for src_node_id, node in enumerate(self.nodes):
                for dest_node_id in node.transitions.values():
                    mat[prev_len+1,src_node_id] += mat[prev_len,dest_node_id]

        # num_strings is float[lengths]
        num_strings_each_length = mat[:,0]

        # probs is float[lengths]
        len_probs = np.array([LENGTH_DISTRIB(k) for k in range(MAX_LENGTH+1)])
        len_probs *= (num_strings_each_length != 0)
        len_probs /= sum(len_probs)

        self.mat = mat
        self.num_strings_each_length = num_strings_each_length
        self.len_probs = len_probs

    def matches(self, s: str) -> bool:
        node = self.nodes[0]
        for c in s:
            if c not in node.transitions:
                return False
            node = self.nodes[node.transitions[c]]
        return node.is_end

    def prob_of(self, s: str) -> float:
        if len(s) > self.MAX_LENGTH:
            raise Exception(f'length of {s=} is too big')
        if not self.matches(s):
            return 0.
        return self.len_probs[len(s)] / self.num_strings_each_length[len(s)]

    def sample(self) -> str:
        length = np.random.choice(self.MAX_LENGTH+1, p=self.len_probs)
        out = ''
        node = self.nodes[0]
        for l in reversed(range(length)):
            chars = np.array(list(node.transitions.keys()))
            probs = np.array(list(self.mat[l,n] for n in
                node.transitions.values())) / self.mat[l+1, node.node_id]
            ch = np.random.choice(chars, p=probs)
            out += ch
            node = self.nodes[node.transitions[ch]]
        return out

