from .index import IndexNode
from .accessor import AccessorNode
from .symbol import SymbolNode
from .base import Node

class AssignmentNode(Node):
    def __init__(self, obj: SymbolNode | AccessorNode, index: IndexNode, value: Node):
        # object on which to assign a value
        self.obj = obj
        #index property name or matric index, Optionaal, if not provided
        #and object is a SymbolNode the property is assigned to the global scope
        self.index = index
        #the value to the assigned
        self.value = value