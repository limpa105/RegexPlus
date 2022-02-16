import math

class VSA:
    nodes: set[str]
    edges: dict[str, dict[str, set[str]]]
    start_node: str
    end_node: str

    def __init__(self, nodes, edges, start_node, end_node):
        self.nodes = nodes
        self.edges = edges
        self.start_node = start_node
        self.end_node = end_node

def possible_regex_tokens(s):
    # FIXME: there should be a better way to represent regex tokens than strings
    yield s # constant string always works
    if s.isdecimal():
        yield "[0-9]+"
    if s.islower():
        yield "[a-z]+"
    if s.isupper():
        yield "[A-Z]+"
    if s.isalpha():
        yield "[a-zA-Z]+"
    if s.isalnum():
        yield "[a-zA-Z0-9]+"
    if s.isspace():
        yield "\\s+"
    # TODO: other tokens (capitalized words, stuff like that)

def mk_vsa(s: str) -> VSA:
    nodes = set(f"v{i}" for i in range(len(s) + 1))
    edges = {}
    for i in range(len(s)):
        outgoing_edges = {}
        for j in range(i+1, len(s)+1):
            outgoing_edges[f"v{j}"] = set(possible_regex_tokens(s[i:j]))
        edges[f"v{i}"] = outgoing_edges
    edges[f"v{len(s)}"] = {}
    start_node = "v0"
    end_node = f"v{len(s)}"
    return VSA(nodes, edges, start_node, end_node)

# Generates a big VSA that you should subsequently get rid of its unreachable
# nodes
# This two-step process could be implemented more efficiently in a single step
def intersect(va: VSA, vb: VSA) -> VSA:
    pairs = [(a, b) for a in va.nodes for b in vb.nodes]
    nodes = set(a + "," + b for a, b in pairs)
    edges = {}
    for a_src, b_src in pairs:
        outgoing_edges = {}
        for a_target, b_target in pairs:
            if a_target not in va.edges[a_src] or b_target not in vb.edges[b_src]:
                continue
            a_edges = va.edges[a_src][a_target]
            b_edges = vb.edges[b_src][b_target]
            outgoing_edges[a_target + "," + b_target] = a_edges.intersection(b_edges)
        edges[a_src + "," + b_src] = outgoing_edges
    start_node = va.start_node + "," + vb.start_node
    end_node = va.end_node + "," + vb.end_node
    return VSA(nodes, edges, start_node, end_node)

# Delete unreachable nodes
def simplify(v: VSA) -> VSA:
    # use DFS to find reachable nodes
    nodes = set()
    def dfs(a):
        if a in nodes:
            return
        nodes.add(a)
        for b, regexes in v.edges[a].items():
            if regexes != {}:
                dfs(b)
    dfs(v.start_node)

    # populate edges
    edges = {}
    for a in nodes:
        outgoing_edges = {}
        for b in nodes:
            if b not in v.edges[a]:
                continue
            outgoing_edges[b] = v.edges[a][b]
        edges[a] = outgoing_edges

    start_node = v.start_node
    end_node = v.end_node
    if end_node not in nodes:
        raise Exception("No possible regex :(")

    return VSA(nodes, edges, start_node, end_node)


def possible_regexes(v: VSA):
    def regexes_starting_at(a):
        if a == v.end_node:
            yield ""
            return
        for b, regexes in v.edges[a].items():
            for r in regexes:
                for rest in regexes_starting_at(b):
                    yield r + rest
    yield from regexes_starting_at(v.start_node)


def wt_of_token(tok: set[str]) -> tuple[int, str]:
    # TODO: come up with good heuristic/good weights
    special_things = {"[0-9]+", "[a-z]+", "[A-Z]+", "[a-zA-Z]+", "[a-zA-Z0-9]+", "\\s+"}
    if "\\s+" in tok:
        return 100, "\\s+"
    elif len(tok.difference(special_things)) > 0:
        # it has a literal string
        s, = tok.difference(special_things)
        return 100, s
    elif "[0-9]+" in tok:
        return 50, "[0-9]+"
    elif "[a-z]+" in tok:
        return 50, "[a-z]+"
    elif "[A-Z]+" in tok:
        return 50, "[A-Z]+"
    elif "[a-zA-Z]+" in tok:
        return 25, "[a-zA-Z]+"
    elif "[a-zA-Z0-9]+" in tok:
        return 10, "[a-zA-Z0-9]+"
    else:
        return -math.inf, ""
        # # set is probably empty
        # assert len(tok) == 0
        # # there is no regex
        # raise Exception('no possible regex')


def get_best_regex(v: VSA) -> str:
    best_from_node = { v.end_node: (0, "") }
    def dfs(a):
        if a in best_from_node:
            return best_from_node[a]
        else:
            cur_best_wt = -math.inf
            cur_best_regex = ""
            for b, regexes in v.edges[a].items():
                wt_of_b, regex_of_b = dfs(b)
                wt, regex = wt_of_token(regexes)
                if wt + wt_of_b > cur_best_wt:
                    cur_best_wt = wt + wt_of_b
                    cur_best_regex = regex + regex_of_b
            best_from_node[a] = cur_best_wt, cur_best_regex
            return cur_best_wt, cur_best_regex
    return dfs(v.start_node)


def main():
    print('Enter examples, leave blank when done')
    inputs = []
    while True:
        i = input('> ')
        if i == "":
            break
        inputs.append(i)

    print("doin' VSA stuff")
    vsa = mk_vsa(inputs[0])
    for s in inputs[1:]:
        vsa = simplify(intersect(vsa, mk_vsa(s)))

    wt, regex = get_best_regex(vsa)
    print(f"Best regex: {regex} (weight {wt})")


if __name__ == '__main__':
    main()
