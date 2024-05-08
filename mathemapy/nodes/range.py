from .base import Node

class RangeNode(Node):
    def __init__(self, start: Node, end: Node, step: Node):
        self.start = start # lower bound 
        self.end = end # upper bound
        self.step = step # optional step, *default 1 [TODO]