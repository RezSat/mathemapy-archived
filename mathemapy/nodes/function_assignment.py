from .base import Node

class FuncttionAssignmentNode(Node):
    def __init__(self, name: str, params: list[str], expr: Node):
        self.name = name # Function Name
        # Array with function parameter names, or an arrway with objects 
        # containing the name and the type of the parameter [ TODO]
        self.params = params
        self.expr = expr # the function expression