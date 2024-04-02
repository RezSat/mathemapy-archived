from .base import Node

class ArrayNode(Node):
    def __init__(self, items) -> None:
        self.items = items