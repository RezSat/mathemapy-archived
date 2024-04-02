from .base import Node

class OperatorNode(Node):
    def __init__(self, op: str, fn: str, args: list, implicit: bool, isPercentage: bool):
        self.op = op  # Operator Name, for example '+'
        self.fn = fn # Function name, for exampel 'add'
        self.args = args # operator arguments
        self.implicit = implicit # is this an implicit multiplications=?
        self.isPercentage = isPercentage # is this an percentage operation?