import re
#oh

operators = {
    '+': 'Add',
    '-': "Subract",
    '/': "Divide",
    '*': "Multiply",
    "^": "Power",
}

tokens = [

        "Number",
        "ComplexNumber",
        "Variable",
        "Function",
        "Equality",
        "SemiColon",
        "Parenthesis",

]

class Token:
    def __init__(self, tok, name, type):
        self.tok = tok
        self.type = type
        self.name = name

    def __repr__(self):
        return f"Token(tok={self.tok}, name={self.name}, type={self.type})"

class Tokenizer:
    def __init__(self, expression):
        self.expression = expression
        self.tokens = []
        
    def tokenize(self):
        equation_pattern = r'='
        semicolon_pattern = r';'
        number_pattern = r'(\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?'
        variable_pattern = r'[a-zA-Z]+'
        operator_pattern = r'[+\-*/^=]'
        function_pattern = r'[a-zA-Z]+\('
        complex_pattern = r'(\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?[+-](\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?[iI]'

        current_token = ""
        while self.expression:
            match = re.match(function_pattern, self.expression)
            if match:
                self.tokens.append( Token(tok=match.group()[:-1], name="Function", type="Operand") )
                self.expression = self.expression[len(match.group()[:-1]):]
                continue

            match = re.match(complex_pattern, self.expression)
            if match:
                self.tokens.append( Token(tok=match.group(), name='ComplexNumber', type='Operand') )
                self.expression = self.expression[len(match.group()):]
                continue

            match = re.match(number_pattern, self.expression)
            if match:
                self.tokens.append( Token(tok=match.group(), name="Number", type="Operand") )
                self.expression = self.expression[len(match.group()):]
                continue
            
            match = re.match(variable_pattern, self.expression)
            if match:
                self.tokens.append( Token(tok=match.group(), name="Variable", type="Operand") )
                self.expression = self.expression[len(match.group()):]
                continue
            
            match = re.match(operator_pattern, self.expression)
            if match:
                self.tokens.append( Token(tok=match.group(), name=operators[match.group()], type="Operator") )
                self.expression = self.expression[len(match.group()):]
                continue
                        
            match = re.match(equation_pattern, self.expression)
            if match:
                self.tokens.append( Token(tok=match.group(), name='Equality', type="Operator") )
                self.expression = self.expression[len(match.group()):]
                continue

            match = re.match(semicolon_pattern, self.expression)
            if match:
                self.tokens.append( Token(tok=match.group(), name="SemiColon", type="Operator") ) 
                self.expression = self.expression[len(match.group()):]
                continue

            if self.expression[0] == '(' or self.expression[0] == ')':
                self.tokens.append( Token(tok=self.expression[0], name="Parenthesis", type="Operator") )
                self.expression = self.expression[1:]
                continue
            
            if self.expression[0] == ';':
                self.tokens.append( Token(tok=self.expresssion[0], name="SemiColon", type="Operator") )
                self.expression = self.expression[1:]
                continue
            
            if self.expression[0].isspace():
                self.expression = self.expression[1:]
                continue
            
            raise ValueError(f"Invalid character: {self.expression[0]}")
        
        return self.tokens


def main():
    expression = "2x+5-(9*7)-5/4 + 2.3e-5 + (4+3.5i) hello();yes();"
    tokenizer = Tokenizer(expression)
    tokens = tokenizer.tokenize()
    print(tokens)
    for i in tokens:
        print(i)


#main()
