from .base import Node

class RelationalNode(Node):
    def __init__(self, conditionals: list, params: list):
        # array of conditional operators used to compare parameters
        self.conditionals = conditionals
        # the parameter that will be compared
        self.params = params