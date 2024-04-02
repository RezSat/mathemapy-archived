from .base import Node

class FunctionNode(Node):
    def __init__(self, fn: Node | str, args: list):
        self.fn = fn
        self.args = args