from .base import Node


class ConstantNode(Node):
    def __init__(self, value: float | int | str):
        self.value = value