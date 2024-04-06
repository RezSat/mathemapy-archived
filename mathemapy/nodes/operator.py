from .base import Node

class OperatorNode(Node):
    def __init__(self, op: str, fn: str, args: list, implicit: bool=False, isPercentage: bool=False):
        self.op = op  # Operator Name, for example '+'
        self.fn = fn # Function name, for exampel 'add'
        self.args = args # operator arguments
        self.implicit = implicit # is this an implicit multiplications=?
        self.isPercentage = isPercentage # is this an percentage operation?

    def __repr__(self) -> str:
        return str(
            {
                "op": self.op,
                "fn": self.fn,
                "args": self.args,
                "implicit": self.implicit,
                "isPercentage": self.isPercentage
            }
        )    