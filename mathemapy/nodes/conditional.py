from .base import Node

class ConditionalNode(Node):
    def __init__(self, condition: Node, true: Node, false: Node) -> None:
        self.condition = condition # condition, must result in  a boolean
        self.true = true # Expression evaluated when condition is true
        self.false = false # Expression evaluated when condition is false