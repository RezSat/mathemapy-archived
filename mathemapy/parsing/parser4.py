from tokenizer2 import Tokenizer, Token, operators

class Operator:
    def __init__(self, tok, left,right):
        self.operator = tok
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Operator(op={self.operator}, left={self.left}, right={self.right})"

class Parenthesis:
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return f"Parenthesis(expr={self.expr})"

class Function:
    def __init__(self, name, arguments, expr=None):
        self.name = name
        self.arguments = arguments
        if expr:
            self.expr = expr
        else:
            self.call = True
    
    def __repr__(self):
        return f"Function(name={self.name}, arguments={self.arguments}, extra={self.expr if self.expr else self.call}"

class Number:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Number(value={self.value})"

class Symbol:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Symbol(value={self.value})"

class ComplexNumber:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "ComplexNumber(value={self.value})"

"""
Factor : Number  Variable ComplexNumber | Individual Elements
Term : Factor | Factor ( * | / ) Factor
Expression : Term ( + | - ) Term

"""
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.i = 0
        self.current_token = self.tokens[self.i]
        self.nestinglevel = 0

    def advance(self):
        eof = Token(tok="\0", name="EOF", type='EOF')
        self.i += 1
        self.current_token = self.tokens[self.i] if self.i < len(self.tokens) else eof
        return self.current_token

    def peek(self, offset=1):
        return self.tokens[self.i + offset]

    def reverse(self, offset=1):
        self.i -= offset
        self.current_token = self.tokens[self.i]
        return self.current_token
    
    def parse(self):
        Equals = Token(tok="=", name="Equals", type="Operator")
        node = self.parse_expression()
        return node

    def parse2(self):
        node = self.parse_expression2()
        return node

    def parse_expression2(self):
        node = self.parse_term()
        
        while self.current_token != None and self.current_token.name in ("Add", "Subtract"):
            op = self.current_token.tok
            self.advance()
            node = Operator(tok=op, left=node, right=self.parse_term())
        return node

    def parse_term(self):
        node = self.parse_factor()
        while self.current_token != None and self.current_token.name in ('Multiply', 'Divide'):
            op = self.current_token.tok
            self.advance()
            node = Operator(tok=op, left=node, right=self.parse_factor())
        return node

    def parse_factor(self):
        node = None
        if self.current_token.tok == "(":
            self.advance()
            expr = self.parse_expression2()
            if self.current_token.tok == ")":
                self.advance()
                node = Parenthesis(expr)

        elif self.current_token.name == "Number":
            node = Number(self.current_token.tok)
            self.advance()
        elif self.current_token.name == "ComplexNumber":
            node = ComplexNumber(self.current_token.tok)
            self.advance()
        elif self.current_token.name == "Variable":
            node = Symbol(self.current_token.tok)
            self.advance()
        
        elif self.current_token.name in ("Add", "Subract"):
            op = self.current_toke.tok
            self.advance()
            node = Operator(tok=op, left=None, right=self.parse_factor())
        return node

    def parse_expression(self):
        node = self.parse_constants()
        if self.current_token.tok in operators :
            op = self.current_token.tok
            self.advance()
            right = self.parse_expression()
            node = Operator(tok=op, left=node, right=right)
        
        if self.current_token.name == "Parenthesis":
            if self.current_token.tok == '(':
                self.nestinglevel += 1
                self.advance()
                node = self.parse_expression()
                if self.current_token.tok == ')':
                    self.nestinglevel -= 1
                    self.advance()
                    return Parenthesis(node)
        return node

    def parse_constants(self):
        node = self.parse_function()
        if self.current_token.name == "Number":
            node = Number(self.current_token.tok)
            self.advance()

        if self.current_token.name == "ComplexNumber":
            node = ComplexNumber(self.current_token.tok)
            self.advance()

        if self.current_token.name == "Variable":
            node = Symbol(self.current_token.tok)
            self.advance()
         
        return node

    def parse_function(self):
        node = None
        if self.current_token.name == "Function":
            fname = self.current_token.tok
            self.advance()
            arguments = self.parse_expression()
            if self.current_token.name == "=":
                self.advance()
                expr = self.parse_expression()
                node = Function( name=fname, arguments=arguments, expr=expr)
            else:
                node = Function( name=fname, arguments=arguments )
        return node

expr = "2*(4+5)/6"
tokenizer = Tokenizer(expr)
tokens = tokenizer.tokenize()
parser = Parser(tokens)
ast = parser.parse2()
print(ast)
