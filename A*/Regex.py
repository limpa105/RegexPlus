from typing import *
import math
import nfa

END_OF_REGEX_PROB = 0.05
ONE_TOKEN_COST = - math.log(1 - END_OF_REGEX_PROB)
OPT_SCALE = 1/3

class Regex:
    def __str__(self) -> str:
        raise NotImplementedError()
    def simplicity_score(self) -> float:
        raise NotImplementedError()
    def specificity_score(self, ex_part: str) -> float:
        raise NotImplementedError()
    def prepend_nfa_to(self, nfa: nfa.Node) -> nfa.Node:
        raise NotImplementedError()
    def opt_simplicity_score(self) -> float:
        raise NotImplementedError()
    def opt_specificty_score(self, ex_part:str) -> float:
        raise NotImplementedError()
    def matches(self, ex: str) -> bool:    
        state = self.prepend_nfa_to(nfa.Node(is_end=True)).epsilon_closure()
        return nfa.matches(state, ex)

class Constant(Regex):
    contents: str

    def __init__(self, contents: str):
        self.contents = contents

    def simplicity_score(self) -> float:
        return TODO 

    def specificity_score(self, ex_part: str) -> float:
        return 
    
    def opt_simplicity_score(self) -> float:
        return super().opt_simplicity_score()



class CharClass(Regex):
    def __init__(self, name: str, simpl_prob: float, options: Set[str]):
        
        self.options = options
        self.name = name
        self.simpl_prob = simpl_prob
        
    
    def __str__(self) -> str:
        return f'[{self.name}]'

    def simplicty_score(self) -> float:
        return - math.log(self.simpl_prob) + ONE_TOKEN_COST

    def specificity_score(self, ex_part: str) -> float:
        if ex_part in self.options:
            return math.log(len(self.options))
        else:
            return math.inf
    
    def opt_simplicty_score(self, ex_part: str) -> float:
        if ex_part in self.options:
            return 1/2 * math.log(len(self.options))
        else:
            return math.inf 

    def opt_specificty_score(self, ex_part: str) -> float:
        size = len(self.contents.options)
        return len(ex_part) * math.log(size + 1) + math.log(size)

    
    def prepend_nfa_to(self, nfa: nfa.Node) -> nfa.Node:
        TODO
        

class RepeatedCharClass(Regex):
    contents: CharClass

    def __init__(self, contents: CharClass):
        self.contents = contents
    
    def __str__(self):
        return str(self.contents) + '+'
    
    def simplicity_score(self) -> float:
        return self.contents.simplicity_score() + math.log(2) 

    def specificity_score(self, ex_part: str) -> float:
        size = len(self.contents.options)
        return len(ex_part) * math.log(size + 1) + math.log(size)

    def opt_simplicty_score(self, ex_part: str) -> float:
        if ex_part in self.options:
            return 1/2 * math.log(len(self.options))
        else:
            return math.inf 
    
    def opt_specificty_score(self, ex_part: str) -> float:
        size = len(self.contents.options)
        return len(ex_part) * math.log(size + 1) + math.log(size)

    '''
    name: str
    simpl_prob: float
    options: Set[str]

    def __init__(self, name: str, simpl_prob: float, options: Set[str]):
        self.name = name
        self.simpl_prob = simpl_prob
        self.options = options

    def __str__(self) -> str:
        return f'[{self.name}]+'

    def simplicity_score(self) -> float:
        return - + math.log(self.simpl_prob) + ONE_TOKEN_COST

    def specificity_score(self, ex_part: str) -> float:
        return len(ex_part) * math.log(len(self.options) + 1) + math.log(len(self.options))

    def prepend_nfa_to(self, nfa: nfa.Node) -> nfa.Node:
        TODO
        '''
    

class Optional(Regex):
    '''TODO: this should have different formulas for stuff depending on what its contents is'''
    contents: Regex

    def __init__(self, contents: Regex):
        self.contents = contents

    def __str__(self) -> str:
        return '(' + str(self.contents) + ')?'
    
    def simplicity_score(self) -> float:
        return self.contents.simplicity_score() + math.log(3)

    def specificity_score(self, ex_part: str) -> float:
        size = len(self.contents.options)
        return len(ex_part) * math.log(size + 1) + math.log(size)



"""
token_probabilities: Dict[str, float] = {
    '[0-9]': 0.095,
    '[0-9]+': 0.0475,
    '[a-z]': 0.095,
    '[a-z]+': 0.0475,
    '[A-Z]': 0.095,
    '[A-Z]+': 0.0475,
    '[a-zA-Z]': 0.01,
    '[a-zA-Z]+': 0.005,
    '[a-zA-Z0-9]': 0.005,
    '[a-zA-Z0-9]+': 0.0025,
}
"""
#[0-9] == 0.095

#[0-9]+ --? [0-9].probaility *1/2



# Example Usage:
#AZ = ("[a-z]")



""" """

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
"""