from .index import IndexNode
from .base import Node

class AccessorNode(Node):
    def __init__(self, obj, index: IndexNode):
        self.obj = obj # object from which to retrieve
        self.index = index #IndexNode contianing ranges