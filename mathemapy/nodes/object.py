from .base import Node

class ObjectNode(Node):
    def __init__(self, properties: dict) -> None:
        self.properties = properties
# @dataclass
# class KeyValueNode(Node):
#     key: str | int
#     value = object

# @dataclass
# class ObjectNode(Node):
#     # holds and object of key/value pairs
#     items: list[KeyValueNode] = field(default_factory=list)

#     def __setitem__ (self, key, value):
#         new_item = KeyValueNode(key=key, value=value)
#         for i, item in enumerate(self.items):
#             if self.items[i] == key:
#                 self.items[i] = new_item
#                 return
#         self.items.append(new_item)

#     def __getitem__ (self, key):
#         for item in self.items:
#             if item.key == key:
#                 return item.value
#         raise KeyError(f"Key {key} not found")
    
#     def __contains__ (self, key):
#         for item in self.items:
#             if item.key == key:
#                 return True
#         return False

            