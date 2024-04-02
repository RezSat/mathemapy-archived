from .base import Node

    
# blocks: list[{'node': Node} | {'node': Node, 'visible': bool}] = field(default_factory=list) # type: ignore

class BlockNode(Node):
    """
    Holds a set with blocks
    an array with blocks, wehre block is 
    constructed as an object with properties block
    which is Node, and visible, which is a boolesn.
    The property visible is optional and 
    is true by default

    ex: a=2;b=3;c=4
    ex: a=2\nb=3
    """
    def __init__(self, blocks: list):
        self.blocks = blocks