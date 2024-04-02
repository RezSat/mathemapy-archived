from mathemapy.parsing.parsing import Parser
from mathemapy.parsing.lexer import tokenizer

tokens = tokenizer("x")
ast = Parser(tokens)
tree = ast.parse()
print(tree)