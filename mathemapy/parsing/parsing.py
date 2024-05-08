#from .tokens import *
from mathemapy.nodes import *
from .lexer import *

"""
TokenInfo:
    exact_type
    type
    string
    start
    end
    string
"""

class Parser:

    """
    Parse the tokens into an object of AbstractSyntaxTree, which then will be evaluated by an Interpreter separately.

    This would have configurations to be passed on later for alternative syntax adjustments for more customization. [ TODO ]
    Other than alternative syntax adjustments, another set of configurations should be supported for adjusting properties while evaluating [TODO]

    """

    def __init__(self, tokens: list):
        self.tokens = tokens
        self.i: int = 1
        self.encoding_info = tokens[0]
        self.nestingLevel: int = 0
        self.conditionalLevel: int = None
        self.current_token = self.tokens[self.i]    

    def peek(self, offset=0):
        return self.tokens[self.i+offset]
    
    def advance(self):
        self.i += 1
        self.current_token = self.tokens[self.i]
        return self.current_token
    
    def reverse(self, offset=0):
        self.i -= offset
        self.current_token = self.tokens[self.i]
        return self.current_token

    def consume(self, token_type):
        if self.peek().type == token_type:
            self.advance()
            return True
        return False
    
    def openParams(self):
        self.nestingLevel += 1

    def closeParams(self):
        self.nestingLevel -= 1

    def parse(self):
        node = self.parseBlock()
        # TODO Error handling cases
        return node

    def parseBlock(self):
        """
        Parse a block with expressions, Expressions can ve separated by a newline
        character '\n' or a semicolon ';'. In case of a semicolom , no output
        of the preceding line is returned.
        """
        node = None
        blocks = []

        if (self.current_token.string != '\n' and 
            self.current_token.exact_type != EOF and 
            self.current_token.exact_type != SEMI):
            node = self.parseAssignment()
            #self.advance()
        while (self.current_token.string == '\n' 
               or self.current_token.exact_type == SEMI):
            if (len(blocks) == 0 and node):
                visible = (self.current_token.exact_type != SEMI)
                blocks.append({'node': node, 'visible': visible})
        
        self.advance()

        if (self.current_token.string != '\n' and 
            self.current_token.exact_type != EOF and 
            self.current_token.exact_type != SEMI):
            node = self.parseAssignment()

            visible = (self.current_token.exact_type != SEMI)
            blocks.append({'node': node, 'visible': visible})
        
        if len(blocks) > 0:
            return BlockNode(blocks=blocks)
        else:
            if not node:
                node = ConstantNode('undefined')
                return node
            
    def parseAssignment(self):
        """
        Assignmet of a function or variable
        - can be a variable like a=3
        - or update an existing variable like matrix(2,3:5) = [6,7,8]
        - defining a function like f(x) = x^2
        """

        name: None 
        args: None
        value: None
        valid: None

        node = self.parseConditional()
        if self.current_token.exact_type == EQUAL:
            if isinstance(node, SymbolNode):
                name = node.name
                self.advance()
                value = self.parseAssignment()
                return AssignmentNode(SymbolNode(name), value=value)
            
        elif isinstance(node, AccessorNode):
            self.advance()
            # pare a matrix subset assignment like A[1,2] = 4
            value = self.parseAssignment()
            return AssignmentNode(node.object,node.index, value)
        
        elif isinstance(node, FunctionNode) and isinstance(node.fn, SymbolNode):
            valid = True
            args = []
            name = node.name
            for arg in node.args:
                if isinstance(arg, SymbolNode):
                    args.append(arg.name)
                else:
                    valid = False
            if valid:
                self.advance()
                value = self.parseAssignment()
            return FuncttionAssignmentNode(name, args, value)
        else:
            return node
        
    def parseConditional(self):
        node = self.parseLogicalOr()
        
        while self.current_token.string == "?":
            self.advance()
            condition = node
            true = self.parseAssignment()
            # TODO Error handle for not having ":" 
            self.advance()
            false = self.parseAssignment()
            node = self.ConditionalNode(node, true, false)
        return node
    
    def parseLogicalOr(self):
        node = self.parseLogicalXor()
        while self.current_token.string == "or":
            self.advance()
            node = OperatorNode('or', 'or', [node, self.parseLogicalXor()])
        return node
    
    def parseLogicalXor(self):
        node = self.parseLogicalAnd()
        while self.current_token.string == "xor":
            self.advance()
            node = OperatorNode('xor', 'xor', [node, self.parseLogicalAnd()])
        return node

    def parseLogicalAnd(self):
        node = self.parseBitwiseOr()
        while self.current_token.string == "and":
            self.advance()
            node = OperatorNode('and', 'and', [node, self.parseBitwiseOr()])
        return node
    
    def parseBitwiseOr(self):
        node = self.parseBitwiseXor()
        while self.current_token.string == "|":
            self.advance()
            node = OperatorNode('|', 'bitOr', [node, self.parseBitwiseXor()])
        return node

    def parseBitwiseXor(self):
        node = self.parseBitwiseAnd()
        while self.current_token.string == "^|":
            self.advance()
            node = OperatorNode('^|', 'bitXor', [node, self.parseBitwiseAnd()])
        return node

    def parseBitwiseAnd(self):
        node = self.parseRelational()
        while self.current_token.string == "&":
            self.advance()
            node = OperatorNode('&', 'bitAnd', [node, self.parseRelational()])
        return node

    def parseRelational(self):
        params = [self.parseShift()]
        conditionals = []

        operators = {
            EQEQUAL: "equal",
            NOTEQUAL: "unequal",
            LESS: "smaller",
            GREATER: "larger",
            LESSEQUAL: "smallerEq",
            GREATEREQUAL: "largerEq", 
        }

        while self.current_token.exact_type in operators:
            conditionals.append(
                {
                    'name': self.current_token.string,
                    'fn': operators[self.current_token.exact_type]
                }
            )
            self.advance()
            params.append(self.parseShift())
        
        if len(params) == 1:
            return params[0]
        elif len(params) == 2:
            return OperatorNode(
                conditionals[0]['name'],
                conditionals[0]['fn'],
                params
            )
        else:
            return RelationalNode(
                [c['fn'] for c in conditionals],
                params=params
            )
    
    def parseShift(self):
        node = self.parseConversion()

        operators = {
            "<<": 'leftShift',
            ">>": 'rightShift',
            ">>>": "rightLogShift"
        }
        
        while self.current_token.string in operators:
            name = self.current_token.string
            fn = operators[name]
            self.advance()
            node = OperatorNode(
                self.current_token.string,
                fn,
                [node, self.parseConversion()]
            )
        return node
    
    def parseConversion(self):
        node = self.parseRange()

        operators = {
           "to": "to",
           "in": "to" # alias for to
        }

        while self.current_token.string in operators:
            name = self.current_token.string
            fn = operators[name]
            self.advance()
            #TODO we need to check for exprssion after the `in` cuz it might be unit `inch` in short form
            #in that case we need to return a OperatorNode("*", 'multiply', [node, SymbolNode('in)], true)

            params = [node, self.parseRange()]
            node = OperatorNode(name,fn , params)
        return node
    
    def parseRange(self):
        if self.current_token.exact_type == COLON:
            node = ConstantNode(1)
        else:
            node = self.parseAddSubtract()

        # TODO build the proper logic to handle the range.
        return node
    
    def parseAddSubtract(self):
        node = self.parseMultiplyDivide()
        operators = {
            "+": "add",
            "-": "subtract"
        }

        while self.current_token.string in operators:
            name = self.current_token.string
            fn = operators[name]
            self.advance()
            rightNode = self.parseMultiplyDivide()

            # if rightNode.isPercentage: # TODO buid isPercentage logic
            #     params = [node, OperatorNode("*", 'multiply', [node, rightNode])]
            # else:
            #     params = [node, rightNode]
            node = OperatorNode(
                name,
                fn,
                [node, rightNode]
            )
        return node

    def parseMultiplyDivide(self):
        node = self.parseImplicitMultiplication()
        last = node

        operators = {
            '*': 'multiply',
            '.*': 'dotMultiply',
            '/': 'divide',
            './': 'dotDivide'
        }

        while True:
            if self.current_token.string in operators:
                name = self.current_token.string
                fn = operators[name]
                self.advance()

                last = self.parseImplicitMultiplication()

                node = OperatorNode(
                    name,
                    fn,
                    [node,last]
                )
            else:
                break

        return node
    
    def parseImplicitMultiplication(self):
        node = self.parseRule2()

        # TODO build the proper logic to handle here
        return node
    
    def parseRule2(self):
        node = self.parsePercentage()
        # TODO built the rule2 logic
        return node
    
    def parsePercentage(self):
        node = self.parseUnary()
        # TODO build the proper logic to handle here
        return node
    
    def parseUnary(self):
        operators = {
            '-': 'unaryMinus',
            '+': 'unaryPlus',
            '~': 'bitNot',
            'not': 'not'
        }

        if self.current_token.string in operators:
            name = self.current_token.string
            fn = operators[name]
            self.advance()
            node = OperatorNode(
                name,
                fn,
                [self.parseUnary()]
            )
            return node
        return self.parsePow()
    
    def parsePow(self):
        node = self.parseLeftHandOperators()
        # TODO is supported .^ add it then
        if self.current_token.string == "^":
            self.advance()
            node = OperatorNode(
                "^",
                "pow",
                [node, self.parseUnary()]
            )
        return node
    
    def parseLeftHandOperators(self):
        node = self.parseCustomNodes()
        operators = {
            "!": "factorial",
            "\\": "ctranspose"
        }

        while self.current_token.string in operators:
            name = self.current_token.string
            fn = operators[name]
            self.advance()
            node = OperatorNode(
                name,
                fn,
                [node]
            )
            node = self.parseAccessors(node)
        return node
    
    def parseCustomNodes(self):
        node = self.parseSymbol()
        # yeah its a pain but built this too
        return node
    
    def parseSymbol(self):

        # things like `mod` , 'to', 'in', 'and' are also consider as symbol
        if self.current_token.type == SYMBOL or (
            self.current_token.type == DELIMITER and
            self.current_token.string in NAMED_DELIMITERS
        ):
            name = self.current_token.string
            self.advance()
            if name in CONSTANTS:
                node = ConstantNode(CONSTANTS[name])
            elif name in NUMBERIC_CONSTANTS:
                node = ConstantNode(name)
            else:
                node = SymbolNode(name)
            
            node = self.parseAccessors(node)
            return node

        return self.parseString()
    
    def parseAccessors(self, node):
        while self.current_token.string in ['.' , '(' , '[']:
            params = []
            if self.current_token.string == '(':
                if isinstance(node, SymbolNode) or isinstance(node, AccessorNode):
                    # function invocation like fn(3,3) or obj.fun(3,4)
                    self.openParams() # implement the function 

                    self.advance()

                    if self.current_token.string != ')':
                        params.append(self.parseAssignment())

                        # parse a list with parameters
                        while self.current_token.string == ',':
                            self.advance()
                            params.append(self.parseAssignment())

                    if self.current_token.string != ')':
                        # handle the error
                        pass
                    self.closeParams() # implement the function
                    self.advance()
                    node = FunctionNode(node, params)
                else:
                    # implicit multiplications like (2+3)(4+5) or sqrt(2)(1+2)
                    # don't parse it here but let it be handled in parseImplicitMultiplication
                    # with correct precendence
                    return node
            elif self.current_token.string == '[':
                self.openParams()
                self.advance()

                if self.current_token.string != ']':
                    params.append(self.parseAssignment())

                    while self.current_token.string == ',':
                        self.advance()
                        params.append(self.parseAssignment())
                if self.current_token.string != ']':
                    # handle the error
                    pass
                self.closeParams()
                self.advance()
                node = AccessorNode(node, IndexNode(params))
            else:
                # dot notatin like obj.prop
                self.advance()
                # error handle here

                params.append(ConstantNode(self.current_token.string))
                self.advance()

                node = AccessorNode(node, IndexNode(params, True))

        return node
    
    def parseString(self):
        if self.current_token.exact_type == DQUOTE or self.current_token.exact_type == SQUOTE:
            string = self.parseStringToken()
            node = ConstantNode(string)
            self.advance() # not sure if this is need so run it and see
            node = self.parseAccessors(node)
            return node
        return self.parseMatrix()
    
    def parseStringToken(self):
        #need logic for parsing strings properly
        pass
    
    def parseMatrix(self):
        params = []
        row = None
        if self.current_token.exact_type == LSQB:
            self.openParams()
            self.advance()
            if self.current_token.exact_type != RSQB:
                row = self.parseRow()
                if self.current_token.exact_type == SEMI:
                    rows = 1
                    params = [row]

                    while self.current_token.exact_type == SEMI:
                        self.advance()

                        if self.current_token.exact_type != RSQB:
                            params.append(self.parseRow())
                            rows += 1
                    if self.cureqnt_token.exact_type != RSQB:
                        # handle the error
                        pass
                    self.closeParams()
                    self.advance()
                    
                    # check if the number of columns matches in all rows

                    # handle errors

                    array = ArrayNode(params)
                else:
                    # handle the error
                    self.closeParams()
                    self.advance()

                    array = row
            else:
                self.closeParams()
                self.advance()
                array = ArrayNode([])

            return self.parseAccersors(array)

        return self.parseObject()
    
    def parseRow(self):
        params = [self.parseAssignment()]

        while self.current_token.exact_type == COMMA:
            self.advance()

            if self.current_token.exact_type != RSQB and self.current_token.exact_type != SEMI:
                params.append(self.parseAssignment())

        return ArrayNode(params)
    
    def parseObject(self):
        if self.current_token.exact_type == LBRACE:
            self.openParams()
            key = None
            properties = {}

            while self.current_token.exact_type == COMMA:
                self.advance()
                if self.current_token.type == SYMBOL or (
                    self.current_token.type == DELIMITER and
                    self.current_token.string in NAMED_DELIMITERS
                ):
                    key = self.current_token.string
                    self.advance()

                if self.current_token.exact_type == COLON:
                    self.advance()
                    properties[key] = self.parseAssignment()
                else:
                    # handle the error
                    pass
            if self.current_token.exact_type != RBRACE:
                # handle the error properly
                print("No } found")
                raise SyntaxError
            self.closeParams()
            self.advance()
            node =  ObjectNode(properties)
            node = self.parseAccessors(node)
            return node
        return self.parseNumber()
    
    def parseNumber(self):
        if self.current_token.type == NUMBER:
            number = self.current_token.string
            #self.advance()
            return ConstantNode(number)
        return self.parseParentheses()
    
    def parseParentheses(self):
        if self.current_token.exact_type == LPAREN:
            self.openParams()
            self.advance()

            node = self.parseAssignment()

            if self.current_token.exact_type != RPAREN:
                # handle the error
                pass
            self.closeParams()
            self.advance()
            node = ParenthesisNode(node)
            node  = self.parseAccessors(node)
            return node
        return self.parseEnd()
    
    def parseEnd(self):
        print(self.current_token)
        if self.current_token.exact_type == EOF:
            raise None
