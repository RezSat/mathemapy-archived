from .base import Node

class SymbolNode(Node):
    def __init__(self, name: str):
        # check whether some name is a valueless unit like "inch" or "cm"
        self.name = name

    def __repr__(self) -> str:
        return str(
            {
                "name": self.name
            }
        )