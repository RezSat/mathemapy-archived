from mathemapy.parsing.parsing import Parser
from mathemapy.parsing.lexer import Tokenizer

math_expressions = [
    "2x+5-(9*7)-5/4 + 23e5",
    "sqrt(4) + 3^2 - log(10) * sin(45)",
    "integral(0, 1, x^2) + derivative(cos(x))",
    "∑(i=1 to n, i^2) - ∫(0, π, sin(x))", # not working abd don't care
    "e^(π*i) = -1",
    "√(a^2 + b^2)" # not working and don't care
]

fn = "<stdin>"
contents = "2+"
tokens = Tokenizer(fn=fn, encoding='utf-8', content=math_expressions[5]).tokenize()
for i in tokens:    print(i)
#ast = Parser(tokens)
#tree = ast.parse()
#print(tree)