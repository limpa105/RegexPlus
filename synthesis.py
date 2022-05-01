import math
import re
import sys
from typing import List, Dict, Tuple, Set
import itertools
import pickle
import numpy as np 
import warnings

from greenery import lego


# ignoring sklearn warnings 
warnings.filterwarnings('ignore')

class VSA:
    num_nodes: int
    edges: List[Dict[int, Tuple[bool, Dict[str, float]]]]
    start_node: int
    end_node: int
    num_inputs: int

    def __init__(self, num_nodes, edges, start_node, end_node, num_inputs):
        self.num_nodes = num_nodes
        self.edges = edges
        self.start_node = start_node
        self.end_node = end_node
        self.num_inputs = num_inputs

    def debug(self):
        print(f"VSA with {self.num_nodes} nodes")
        print(f"start: {self.start_node}, end: {self.end_node}")
        for i, e in enumerate(self.edges):
            print(f"  {i}: {e}")

def add_backslashes(s):
    return ''.join([disambiguate_char(c) for c in s])

def disambiguate_char(c):
    return '\\'*(c in '[]()\{\}*+?|$.\\') + c  # no ^s, greenery doesn't like escaping them

regparts_base = [r'[0-9]', r'[a-z]', r'[A-Z]', r'[a-zA-Z]', r'[a-zA-Z0-9]']
reg_strs = []
for part in regparts_base:
    reg_strs.append(part)
    reg_strs.append(part+'+')

def possible_regex_tokens(s):
    # FIXME: there should be a better way to represent regex tokens than strings
    yield add_backslashes(s) # constant string always works
    for reg_str in reg_strs:
        if (re.compile(reg_str+r'$')).match(s):
            yield reg_str
    # TODO: other tokens (capitalized words, stuff like that)

char_class_sizes: Dict[str, float] = {
    '[0-9]': 10,
    '[0-9]+': 10,
    '[a-z]': 26,
    '[a-z]+': 26,
    '[A-Z]': 26,
    '[A-Z]+': 26,
    '[a-zA-Z]': 52,
    '[a-zA-Z]+': 52,
    '[a-zA-Z0-9]': 62,
    '[a-zA-Z0-9]+': 62,
}
def prob_doesnt_take_optional_ish(regex: str) -> float:
    if regex in char_class_sizes:
        return 1/char_class_sizes[regex]
    else:
        return 1
def original_prob(regex: str, s: str) -> float:
    if regex in char_class_sizes:
        # It is a character class, or (character class)+
        size = char_class_sizes[regex]
        if regex[-1] == '+':
            # It is (character class)+
            return 1/size * (1/(size + 1)) ** len(s)
        else:
            # It is a plain character class
            return 1/size
    else:
        # It is a literal string
        return 1.0

def mk_vsa(s: str) -> VSA:
    num_nodes = len(s) + 1
    edges = []
    for i in range(len(s)):
        outgoing_edges = {}
        for j in range(i+1, len(s)+1):
            outgoing_edges[j] = False, { regex: original_prob(regex, s[i:j]) for
                    regex in possible_regex_tokens(s[i:j]) }
        edges.append(outgoing_edges)
    edges.append({})
    start_node = 0
    end_node = len(s)
    num_inputs = 1
    return VSA(num_nodes, edges, start_node, end_node, num_inputs)

def change_to_optional(regex: str, old_prob: float, num_inputs: int) -> float:
    '''Calculate the new probability from making an edge optional'''
    if regex in char_class_sizes:
        # It is a character class, or (character class)+
        size = char_class_sizes[regex]
        P = old_prob * (size / (size + 1)) ** num_inputs
    else:
        # It is a literal string
        assert old_prob == 1.
        P = 0.5 ** num_inputs
    return P


# Intersect two VSAs, only retaining reachable nodes, and deduplicating the VSA
# with congruence closure
def intersect(va: VSA, vb: VSA) -> VSA:
    edges = []
    def new_node_id(es):
        node_id = len(edges)
        edges.append(es)
        return node_id
    memo = {}
    def dfs(a, b):
        if (a, b) in memo:
            return memo[a, b]
        if a == va.end_node and b == vb.end_node:
            node_id = new_node_id({})
            memo[a, b] = True, node_id
            return True, node_id

        children = {}
        def process_edge(a_target, b_target, edge):
            can_reach_end, target = dfs(a_target, b_target)
            if can_reach_end:
                children[target] = edge

        # regular edges
        for a_target, a_edge in va.edges[a].items():
            for b_target, b_edge in vb.edges[b].items():
                is_opt = a_edge[0] or b_edge[0]
                ae = a_edge[1]
                be = b_edge[1]
                e = { regex: ae[regex] * be[regex]
                        for regex in ae.keys() & be.keys() }
                if len(e) > 0:
                    process_edge(a_target, b_target, (is_opt, e))
        if USE_OPTIONALS:
            # optional edges where a is constant
            for b_target, b_edge in vb.edges[b].items():
                new_probs = {
                        regex: p * prob_doesnt_take_optional_ish(regex)**va.num_inputs
                        for regex, p in b_edge[1].items() }
                process_edge(a, b_target, (True, new_probs))
            # optional edges where b is constant
            for a_target, a_edge in va.edges[a].items():
                new_probs = {
                        regex: p * prob_doesnt_take_optional_ish(regex)**vb.num_inputs
                        for regex, p in a_edge[1].items() }
                process_edge(a_target, b, (True, new_probs))

        if len(children) == 0:
            memo[a, b] = False, -1
            return False, -1
        node_id = new_node_id(children)
        memo[a, b] = True, node_id
        return True, node_id

    can_reach_end, start_node = dfs(va.start_node, vb.start_node)
    if not can_reach_end:
        raise Exception("No possible regex :(")
    num_nodes = len(edges)
    end_node = memo[va.end_node, vb.end_node][1]
    num_inputs = va.num_inputs + vb.num_inputs

    return VSA(num_nodes, edges, start_node, end_node, num_inputs)


# Enumerate *all* possible regexes.  We don't use this function
def all_the_possible_regexes(v: VSA) -> List[str]:
    def regexes_starting_at(a):
        if a == v.end_node:
            yield ""
            return
        for b, regexes in v.edges[a].items():
            for r in regexes:
                for rest in regexes_starting_at(b):
                    yield r + rest
    yield from regexes_starting_at(v.start_node)


# this determines parse order... yeck
token_probabilities: Dict[str, float] = {
    '[0-9]': 0.06,
    '[0-9]+': 0.03,
    '[a-z]': 0.06,
    '[a-z]+': 0.03,
    '[A-Z]': 0.06,
    '[A-Z]+': 0.03,
    '[a-zA-Z]': 0.06,
    '[a-zA-Z]+': 0.03,
    '[a-zA-Z0-9]': 0.06,
    '[a-zA-Z0-9]+': 0.03,
}

def simplicity_prob(is_opt: bool, regex: str) -> float:
    if regex in token_probabilities:
        if is_opt:
            return 1/3 * token_probabilities[regex]
        else:
            return token_probabilities[regex]
    else:
        # It is a literal string
        p = (1/96)**(len(regex)+1)
        if is_opt:
            return p * 0.30
        else:
            return p * 0.10

def specificity_prob(num_inputs: int, is_opt: bool, old_prob: float, regex: str) -> float:
    if is_opt:
        return change_to_optional(regex, old_prob, num_inputs)
    else:
        return old_prob

def prob_of_token(num_inputs: int, is_opt: bool, tok: Dict[str, float]) -> Tuple[float, str]:
    def whole_prob(regex, p):
        return (simplicity_prob(is_opt, regex)
                * specificity_prob(num_inputs, is_opt, p, regex))

    return max((whole_prob(regex, p), regex) for regex, p in tok.items())

# for normalizing a regex (R : lego.pattern), use R.reduce()

def get_best_regexes(v: VSA, k=5) -> List[Tuple[float, str]]:
    '''Return the top k regexes. By default k = 5'''
    best_from_node = { v.end_node: [(0, "")] }
    def dfs(a):
        if a in best_from_node:
            return best_from_node[a]
        else:
            # make a list of all possibilities
            cur_best = set()
            for b, regexes in v.edges[a].items():
                prob, regex = prob_of_token(v.num_inputs, regexes[0], regexes[1])
                if regexes[0]:
                    # Correct the optionals probability
                    # prob = change_to_optional(regex, prob, v.num_inputs)
                    wt = - math.log(prob)
                    for wt_of_b, regex_of_b in dfs(b):
                        cur_best.add((wt + wt_of_b, '(' + regex + ')?' + regex_of_b))
                else:
                    wt = - math.log(prob)
                    for wt_of_b, regex_of_b in dfs(b):
                        cur_best.add((wt + wt_of_b, regex + regex_of_b))
            # get only the top k
            best = sorted(cur_best)[:k]
            best_from_node[a] = best
            return best
    return dfs(v.start_node)


def synthesize(inputs):
    
    ## getting features from the examples 
    ## getting the variance
    #std = np.asarray([ len(j) for j in inputs]).std() 
    ## getting the mean
    #mean_len = np.asarray([ len(j) for j in inputs]).mean()
    #dif = np.asarray([len(j) for j in inputs]).max() - np.asarray([len(j) 
    #        for j in inputs]).min()
    ## length of shared constants 
    #ans = inputs[0]
    #for j in inputs:
    #    ans = set(j).intersection(ans)
    #shared_count = len(ans)

    #from numpy import linalg as LA
    #def cosine_sim(v1,v2):
    #    v1 = np.asarray(v1)
    #    v2 = np.asarray(v2)
    #    index = np.argmax([len(v1), len(v2)])
    #    if index==0:
    #        v2 = np.pad(v2,(0,len(v1)-len(v2)),'constant')
    #    else:
    #        v1 = np.pad(v1,(0,len(v2)-len(v1)),'constant')
    #    return np.dot(v1,v2)/(LA.norm(v1)*LA.norm(v2))

    #test = [list(j) for j in inputs]
    #test =  [[ord(i) for i in j] for j in test]
    #test = [cosine_sim(i, test[0]) for i in test]
    #cosine_sim = np.mean(test[1:])

    ## loading the two models from files
    #constants_model = pickle.load(open("constants_model.sav", 'rb'))
    #optionals_model = pickle.load(open("optionals_model.sav", 'rb'))
    #if cosine_sim == np.inf or cosine_sim == -np.inf or math.isnan(cosine_sim):
    #    cosine_sim = 0

    ## predicting the probability that the regex will include a constant 
    #const_prob = constants_model.predict_proba([[std,shared_count,dif, cosine_sim]])[0][1]

    ##predicting the probability that the regex will NOT include an optional
    #opt_prob = optionals_model.predict_proba([[shared_count,std,cosine_sim]])[0][0]

    #print("Probability of a constant", round(const_prob, 3))
    #print("Probability of NOT an optional", round(opt_prob, 3))


    print("Making and intersecting VSAs...")
    vsa = mk_vsa(inputs[0])
    # print("there are %d nodes" % vsa.num_nodes)
    for s in inputs[1:]:
        vsa = intersect(vsa, mk_vsa(s))
    print("there are %d nodes" % vsa.num_nodes)

    print("Doing DFS...")
    regs_with_dupes = get_best_regexes(vsa, k=20)
    # print(f"Best regex: {regex} (weight {wt})")

    print("Simplifying and ranking...")
    # remove duplicates from the list
    regexes = {}
    for (score, raw_reg) in regs_with_dupes:
        reg = lego.parse(raw_reg).reduce()
        if reg not in regexes:
            regexes[reg] = score
        regexes[reg] = min(score, regexes[reg])
    # print(regexes)

    best_regs = sorted([(y,x) for x,y in regexes.items()], key=lambda pair: pair[0])[:5]

    return best_regs

USE_OPTIONALS = False  # lol
if __name__ == '__main__':
    print('Enter examples, leave blank when done')
    inputs = []
    while True:
        i = input('> ')
        if i == "":
            break
        inputs.append(i)

    # Only enable regexes with `?` if the -q flag is used
    # global USE_OPTIONALS
    if len(sys.argv) > 1:
        if sys.argv[1] == '-q':
            USE_OPTIONALS = True

    print("doin' VSA stuff")

    for i, (wt, regex) in enumerate(synthesize(inputs)):
        print(f"  {i+1}. ({wt}, {regex})")
