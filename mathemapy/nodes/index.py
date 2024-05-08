from .base import Node

class IndexNode(Node):
    def __init__(self, dimensions: list, dotNotation: bool) -> None:
        self.dimensions = dimensions
        self.dotNotation = dotNotation