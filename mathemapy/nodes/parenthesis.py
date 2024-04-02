from .base import Node

class ParenthesisNode(Node):
    # this node describes manual parenthesis from source
    def __init__(self, content: Node):
        self.content = content