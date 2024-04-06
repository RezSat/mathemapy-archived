from .base import Node


class ConstantNode(Node):
    def __init__(self, value: float | int | str):
        self.value = value

    def __repr__(self) -> str:
        return str(
            {
                "value": self.value
            }
        )