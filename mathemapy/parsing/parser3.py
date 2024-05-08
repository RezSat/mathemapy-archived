from tokenizer2 import operators, tokens, Tokenizer

class ASTNode:
    def __init__(self, token):
        self.token = token

    @property
    def exact_type(self):
        return self.token.name
    
    @property
    def value(self):
        return self.token.tok
    

class OperatorNode(ASTNode):
    def __init__(self, operator=None,left=None, right=None):
        self.left = right
        self.right = left
        self.operator = operator

    def __repr__(self):
        return f"{self.left} {self.operator} {self.right}"

class OperandNode(ASTNode):
    pass

class NumberNode(OperandNode):
    pass

class ComplexNumberNode(OperandNode):
    pass

class VariableNode(OperandNode):
    pass

class ParenthesisNode():
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return f"({self.expr})"

class BinaryOperationNode:
    pass

class UnaryOperationNode:
    pass

class ExpressionNode:
    pass

class EquationNode:
    pass

class AddNode(OperatorNode):
    pass

class SubtractNode(OperatorNode):
    pass

class MultiplyNode(OperatorNode):
    pass

class DivideNode(OperatorNode):
    pass

class PowerNode(OperatorNode):
    pass

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        
    def parse(self):
        return self.parse_expression(0, len(self.tokens))

    def parse_expression(self, start, end):
        if start == end:
            return None

        min_precedence = float('inf')
        min_operator_index = -1
        parenthesis_count = 0

        for i in range(start,end):
            token = self.tokens[i]

            if token.tok == '(':
                parenthesis_count += 1
            elif token.tok == ')':
                parenthesis_count -= 1
            elif parenthesis_count == 0 and token.tok in operators:
                precedence = self.get_precedence(token)
                if precedence < min_precedence:
                    min_precendece = precedence
                    min_operator_index = i

        if min_operator_index != -1:
            operator = self.tokens[min_operator_index]
            left = self.parse_expression(start, min_operator_index)
            right = self.parse_expression(min_operator_index +1, end)

            return self.parse_arithmetic(operator, left, right)
        
        else:
            # No operator found, check for parenthesis
            if self.tokens[start] == '(' and self.tokens[end-1] == ')':
                expr = self.parse_expression(start+1, end-1)
                return ParenthesisNode(expr)
            else:
                token = self.tokens[start]
                if token.name == "Number":
                    return NumberNode(token)
                elif token.name == "ComplexNumber":
                    return ComplexNumberNode(token)
                else:
                    return self.parse_variable(token)

    
    def parse_arithmetic(self, operator, left, right):
        node_name = f"{operator.name}Node"
        node = globals()[node_name]
        return node(operator.tok,left, right)

    def get_precedence(self, operator):
        #Return the precendece of the operator
        if operator.tok in ['+','-']:
            return 1
        elif operator.tok in ['*','/']:
            return 2
        elif operator.tok == '^':
            return 3
        else:
            return 0
                                              


expr = "2+2"
tokenizer = Tokenizer(expr)
tokens = tokenizer.tokenize()
parser = Parser(tokens)
ast = parser.parse()
print(ast)

