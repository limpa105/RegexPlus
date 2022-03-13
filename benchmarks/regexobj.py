class RegexObj:
    text: str
    size: int
    kind: str

    def __init__(self, text, size):
        self.text = text
        self.size = size
        self.kind = "token"  # only allow initting tokens here
    
    def __str__(self):
        # we want print(obj) to show the regex as text
        return self.text
    
class ConcatObj(RegexObj):
    left: RegexObj
    right: RegexObj

    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.depth = max(left.depth, right.depth) + 1
        self.size = left.size + right.size
        self.kind = "concat"

    def __str__(self):
        left_txt = str(self.left)
        if self.left.kind == "concat":
            left_txt = left_txt[1:-1]
        right_txt = str(self.right)
        if self.right.kind == "concat":
            right_txt = right_txt[1:-1]
        return '('+left_txt+right_txt+')'

class ChoiceObj(RegexObj):
    left: RegexObj
    right: RegexObj

    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.size = left.size + right.size
        self.kind = "choice"

    def __str__(self):
        left_txt = str(self.left)
        if self.left.kind in ["choice", "concat"]:
            left_txt = left_txt[1:-1]
        right_txt = str(self.right)
        if self.right.kind in ["choice", "concat"]:
            right_txt = right_txt[1:-1]
        return '('+left_txt+'|'+right_txt+')'

class VariableNumObject(RegexObj):
    inner: RegexObj

    def __new__(self, _):
        raise Exception("VariableNumObject should never be created")

    def __init__(self, inner):
        self.inner = inner
        self.size = inner.size + 1

class OptionObj(VariableNumObject):
    def __init__(self, inner):
        super().__init__(inner)
        self.kind = "optional"
    
    def __str__(self):
        return str(self.inner)+'?'

class PlusObj(VariableNumObject):
    def __init__(self, inner):
        super().__init__(inner)
        self.kind = "plus"

    def __str__(self):
        return str(self.inner)+'+'

class StarObj(VariableNumObject):
    def __init__(self, inner):
        super().__init__(inner)
        """
        Star objs will only be constructed by reducing
        R+? or R?+ -> R*, but that reduction shouldn't
        change the size
        """
        self.size -= 1
        self.kind = "star"
    
    def __str__(self):
        return str(self.inner)+'*'

class CountObj(RegexObj):
    def __init__(self, inner, num):
        self.inner = inner
        self.num = num
        self.kind = "count"
        self.size = inner.size + 1
    
    def __str__(self):
        return str(self.inner)+'{'+str(self.num)+'}'