from tokenizer import *
from mathemapy.nodes import *

EOF = 0
NUMBER = 1
SYMBOL = 2
OPERATOR = 3
WORD = 4
STRING = 5
UNKNOWN = 6

#Nodes
class Node:
    pass
class BlockNode(Node):
    def __init__(self, *arg) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"BockNode( {self.arg} )"

class AssignmentNode(Node):
    def __init__(self, *arg) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"AssignmentNode( {self.arg} )"

class ExpressionNode(Node):
    def __init__(self, *arg) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"ExpressionNode( {self.arg} )"
    
class EquationNode(Node):
    def __init__(self, *arg) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"EquationNode( {self.arg} )"
    
class InequalityNode(Node):
    def __init__(self, *arg) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"InequalityNode( {self.arg} )"
    
class IfStatementNode(Node):
    def __init__(self, *arg) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"IfStatementNode( {self.arg} )"

class ForLoopNode(Node):
    def __init__(self, *arg) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"ForLoopNode( {self.arg} )"

class WhileLoopNode(Node):
    def __init__(self, *arg) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"WhileLoopNode( {self.arg} )"

class FunctionDefinitionNode(Node):
    def __init__(self, *arg) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"FunctionDefinitionNode( {self.arg} )"

class BinaryOperationNode(Node):
    def __init__(self, *arg) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"BinaryOperationNode( {self.arg} )"
    
class UnaryOperationNode(Node):
    def __init__(self, *arg) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"UnaryOperationNode( {self.arg} )"
    
class AccessorNode(Node):
    def __init__(self, *arg) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"AccessorNode( {self.arg} )"
    
class ParenthesisNode(Node):
    def __init__(self, *arg) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"ParenthesisNode( {self.arg} )"
    
class ListNode(Node):
    def __init__(self, *arg) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"ListNode( {self.arg} )"
class DictionaryNode(Node):
    def __init__(self, *arg) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"DictionaryNode( {self.arg} )"
    
class FunctionCallNode(Node):
    def __init__(self, *arg) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"FunctionCallNode( {self.arg} )"
    
class SymbolNode(Node):
    def __init__(self, *arg) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"SymbolNode( {self.arg} )"
    
class NumberNode(Node):
    def __init__(self, *arg) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"NumberNode( {self.arg} )"
    
class StringNode(Node):
    def __init__(self, *arg) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"StringNode( {self.arg} )"

 
class Parser:
    def __init__(self, tokens) -> None:
        self.tokens = tokens
        self.i = 0
        self.nestingLevel = 0
        self.conditionalLevel = None
        self.current_token = self.tokens[self.i]

    def advance(self):
        self.i += 1
        self.current_token = self.tokens[self.i]

    def reverse(self, offset=0):
        self.i -= offset
        self.current_token = self.tokens[self.i]

    def peek(self, offset=0):
        return self.tokens[self.i + offset]
    
    def parse(self):
        node = self.parse_block()
        return node 
    
    def parse_block(self):
        statements = []
        # 0: EOF 
        while self.current_token.type != EOF:
            statement = self.parse_statement()
            if statement:
                statements.append(statement)
        return BlockNode(statements)
    
    def parse_statement(self):

        if self.current_token.type == NUMBER:
            # could be an ExpressionNode
            # could be an EquationNode
            # could be an InequalityNode
            left_expr = self.parse_expression()
            if self.current_token.string == '=':
                # could be an EquationNode
                self.advance()
                right_expr = self.parse_expression()
                self.advance()
                return EquationNode(left_expr, '=', right_expr)

            elif self.current_token.string in ['<', '>', '<=', '>=', '!=']:
                # could be an InequalityNode
                operator = self.current_token.string
                self.advance()
                right_expr = self.parse_expression()
                self.advance()
                return InequalityNode(left_expr, operator, right_expr)
            else:
                return left_expr
                

        if self.current_token.type == SYMBOL:
            # could be an AssignmentNode
            # could be an EquationNode
            # could be an ExpressionNode
            # could be an InequalityNode
            # could be an FunctionDefinitionNode
            pass

        if self.current_token.type == OPERATOR:
            # could be an ExpressionNode
            # could be an EquationNode
            # could be an InequalityNode
            pass

        if self.current_token.type == WORD:
            # could be an Assignmentnode
            # could be an IfStatementNode
            # could be an ForLoopNode
            # could be an WhileLoopNode
            # could be an FunctionDefinitionNode

            # but could also be an ExpressionNode
            # could be an EquationNode
            # could be an InequalityNode
            pass
        if self.current_token == STRING:
            # could be an ExpressionNode only
            pass

        if self.current_token == UNKNOWN:
            print('Error Unknown token found:', self.current_token.string)

    def parse_expression(self):
        if self.current_token.type == NUMBER:
            # could be NumberNode
            # could be BinaryNode
            pass

        if self.current_token.type == SYMBOL:
            # could be SymbolNode
            # could be BinaryNode
            # could be AccessorNode
            # could be FunctionCall
            pass

        if self.current_token.type == OPERATOR:
            # could be UnaryNode
            # could be BinaryNode
            # could be ParenthesisNode
            pass

        if self.current_token.type == WORD:
            # could be SymbolNode
            # could be List
            # could be AccessorNode
            # could be FunctionCall
            pass

        def parse_assignment(self):
            pass

        def parse_equation_inequality(self):
            pass

        def parse_if_statement(self):
            pass

        def parse_for_loop(self):
            pass

        def parse_while_loop(self):
            pass

        def parse_while_loop(self):
            pass

        def parse_function_definition(self):
            pass

        def parse_binary_operation(self):
            pass

        def parse_unary_operation(self):
            pass

        def parse_parenthesis(self):
            pass

        def parse_list(self):
            pass

        def parse_dictionary(self):
            pass

        def parse_dictionary(self):
            pass

        def parse_accessor_and_calls(self):
            pass

        def parse_symbol(self):
            pass

        def parse_number(self):
            pass

        



tokens = Tokenizer("2x+5-(8+7)/4t").tokenize()