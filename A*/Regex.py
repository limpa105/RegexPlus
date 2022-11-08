from typing import *
import math
import nfa
import functools, dataclasses, string

END_OF_REGEX_PROB = 0.05
ONE_TOKEN_COST = - math.log(1 - END_OF_REGEX_PROB)

class Regex:
    def __str__(self) -> str:
        raise NotImplementedError()
    def simplicity_score(self) -> float:
        raise NotImplementedError()
    def specificity_score(self, ex_part: str) -> float:
        raise NotImplementedError()
    def opt_simplicity_score(self) -> float:
        raise NotImplementedError()
    def opt_specificity_score(self, ex_part: str) -> float:
        raise NotImplementedError()

    @functools.cache
    def prepend_nfa_to(self, node: nfa.Node) -> nfa.Node:
        raise NotImplementedError()

    def matches(self, ex: str) -> bool:    
        state = self.prepend_nfa_to(nfa.Node(is_end=True)).epsilon_closure()
        return nfa.matches(state, ex)

@dataclasses.dataclass(frozen=True)
class Constant(Regex):
    contents: str

    def __str__(self) -> str:
        special = '()[]\\+*?'
        return ''.join(('\\' + c if c in special else c) for c in self.contents)

    def simplicity_score(self) -> float:
        PRINTABLE_CHARS = 95
        return ONE_TOKEN_COST + -math.log(0.3) \
            + math.log(PRINTABLE_CHARS + 1) * len(self.contents) \
                + math.log(PRINTABLE_CHARS)

    def specificity_score(self, ex_part: str) -> float:
        if ex_part == self.contents:
            return 0.
        else:
            return math.inf
    
    def opt_simplicity_score(self) -> float:
        return math.log(3) + self.simplicity_score()

    def opt_specificity_score(self, ex_part: str) -> float:
        if ex_part == self.contents or ex_part == '':
            return math.log(2)
        else:
            return math.inf

    def prepend_nfa_to(self, node: nfa.Node) -> nfa.Node:
        for char in reversed(self.contents):
            node = nfa.Node(transitions={char: node})
        return node


# NOTE [Mark]: eq=False makes it default to object equality. This is a lot
# faster, and works because we only ever create a few instances of these classes
# and then re-use them.

@dataclasses.dataclass(frozen=True, eq=False)
class CharClass(Regex):
    name: str
    simpl_prob: float
    options: FrozenSet[str]

    def __str__(self) -> str:
        return f'[{self.name}]'

    def simplicity_score(self) -> float:
        return - math.log(self.simpl_prob) + ONE_TOKEN_COST

    def specificity_score(self, ex_part: str) -> float:
        if ex_part in self.options:
            return math.log(len(self.options))
        else:
            return math.inf

    def opt_simplicity_score(self) -> float:
        return math.log(3) + self.simplicity_score()

    def opt_specificity_score(self, ex_part: str) -> float:
        if ex_part in self.options or ex_part == '':
            return math.log(len(self.options) + 1)
        else:
            return math.inf
    
    def prepend_nfa_to(self, node: nfa.Node) -> nfa.Node:
        return nfa.Node(transitions={char: node for char in self.options})

@dataclasses.dataclass(frozen=True, eq=False)
class RepeatedCharClass(Regex):
    contents: CharClass

    def __str__(self):
        return str(self.contents) + '+'

    def simplicity_score(self) -> float:
        return self.contents.simplicity_score() + math.log(2) 

    def specificity_score(self, ex_part: str) -> float:
        if all(c in self.contents.options for c in ex_part) and ex_part != '':
            size = len(self.contents.options)
            return len(ex_part) * math.log(size + 1) + math.log(size)
        else:
            return math.inf

    def opt_simplicity_score(self) -> float:
        return math.log(3) + self.simplicity_score()

    def opt_specificity_score(self, ex_part: str) -> float:
        if all(c in self.contents.options for c in ex_part):
            size = len(self.contents.options)
            return (len(ex_part) + 1) * math.log(size + 1)
        else:
            return math.inf

    def prepend_nfa_to(self, node: nfa.Node) -> nfa.Node:
        temp = nfa.Node(epsilon_transitions={node})
        temp.transitions = {char: temp for char in self.contents.options}
        return nfa.Node(transitions={char: temp for char in self.contents.options})

@dataclasses.dataclass(frozen=True)
class Optional(Regex):
    contents: Regex

    def __str__(self) -> str:
        return '(' + str(self.contents) + ')?'
    
    def simplicity_score(self) -> float:
        return self.contents.opt_simplicity_score()

    def specificity_score(self, ex_part: str) -> float:
        return self.contents.opt_specificity_score(ex_part)

    def prepend_nfa_to(self, node: nfa.Node) -> nfa.Node:
        result = self.contents.prepend_nfa_to(node)
        result.epsilon_transitions.add(node)
        return result


CHAR_CLASSES: List[CharClass] = [
    CharClass('a-z', 0.095, frozenset(string.ascii_lowercase)),
    CharClass('0-9', 0.095, frozenset(string.digits)),
    CharClass('A-Z', 0.095, frozenset(string.ascii_uppercase)),
    CharClass('a-zA-Z', 0.01, frozenset(string.ascii_letters)),
    CharClass('a-zA-Z0-9', 0.005, frozenset(string.ascii_letters + string.digits)),
]
REPEATED_CHAR_CLASSES: List[RepeatedCharClass] \
        = [RepeatedCharClass(cc) for cc in CHAR_CLASSES]

def atomic_regexes_matching(text: str) -> Iterable[Regex]:
    '''Does not include optional things, because there are too many'''
    # Constant
    yield Constant(text)
    # Char classes and repeated char classes
    for cc in CHAR_CLASSES + REPEATED_CHAR_CLASSES:
        if cc.matches(text):
            yield cc
