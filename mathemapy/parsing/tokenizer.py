import tabulate
import unicodedata

#TOKEN TYPES
TOKEN_TYPES = {
    0: 'EOF',
    1: 'NUMBER',
    2: 'SYMBOL',
    3: 'OPERATOR',
    4: 'WORD',
    5: 'STRING',
    6: 'UNKNOWN',
}

#OPERATORS
OPERATORS = {

    '(': 'LPAREN',
    ')': 'RPAREN',
    '[': 'LBRACKET',
    ']': 'RBRACKET',
    '{': 'LBRACE',
    '}': 'RBRACE',
    ',': 'COMMA',
    ';': 'SEMICOLON',
    ':': 'COLON',

    '+': 'PLUS',
    '-': 'MINUS',
    '*': 'STAR',
    '/': 'SLASH',
    '%': 'PERCENT',
    '^': 'CARET',
    '!': 'EXCLAMATION',

    '=': 'EQUAL',
    '==': 'EQEQUAL',
    '!=': 'NOTEQUAL',
    '<': 'LESS',
    '<=': 'LESSEQUAL',
    '>': 'GREATER',
    '>=': 'GREATEREQUAL',

    '.': 'DOT',

}

class TokenInfo:
    def __init__(self, type, string, exact_type=None, start=None, end=None) -> None:
        self.type = type
        self.exact_type = exact_type if exact_type else TOKEN_TYPES[self.type]
        self.string = string

        if start:
            self.start = start.copy()
            self.end = start.copy()
            self.end.advance()

        if end:
            self.end = end.copy()

    def __repr__(self) -> str:
        return f"start:{self.start} end:{self.end}\t\t{TOKEN_TYPES[self.type]}\t\t'{self.string}'"

class Position:
    def __init__(self, line, column, index, content):
        self.line = line
        self.column = column
        self.index = index
        self.content = content

    def copy(self):
        return Position(self.line, self.column, self.index, self.content)

    def advance(self):
        if self.index < len(self.content):
            if self.content[self.index] == '\n':
                self.line += 1
                self.column = 0
            else:
                self.column += 1
        self.index += 1

    def __repr__(self) -> str:
        return f"{self.line};{self.column}"

class Tokenizer:
    def __init__(self, source) -> None:
        self.source = source
        self.pos = Position(1, 0, -1, source)
        self.c = None
        self.next()

    def next(self):
        self.pos.advance()
        self.c = self.source[self.pos.index] if self.pos.index < len(self.source) else None
        

    def peek(self, offset=0):
        return self.source[self.pos.index+offset] if ((self.pos.index+offset) < len(self.source)) else ''
    
    def tokenize(self):
        tokens = []
        while self.c != None or self.c == '\0':
            if self.c.isspace():
                self.next()
                
            elif unicodedata.category(self.c).startswith('Ll') or unicodedata.category(self.c).startswith('Lu') or unicodedata.category(self.c).startswith('GREEK') or unicodedata.category(self.c).startswith('Latin'):
                start = self.pos.copy()
                char = self.c
                self.next()
                if (self.c == None or self.c == '\0'): break
                while unicodedata.category(self.c).startswith('Ll') or unicodedata.category(self.c).startswith('Lu') or  unicodedata.category(self.c).startswith('GREEK') or unicodedata.category(self.c).startswith('Latin'):
                    char += self.c
                    self.next()
                    if (self.c == None or self.c == '\0'): break

                if len(char) == 1:
                    tokens.append(TokenInfo(2, char, start=start, end=self.pos.copy()))
                else:
                    tokens.append(TokenInfo(4, char, start=start, end=self.pos.copy()))
            
            elif self.c.isdigit() or (self.c == '.'):
                start = self.pos.copy()
                if (char := self.c+self.peek(1)) in ['0b', '0o', '0x']:
                    self.next(); self.next()
                    if self.c == None or self.c == "\0": break

                    while self.c.isalnum():
                        char += self.c
                        self.next()
                        if self.c == None or self.c == "\0": break

                    if self.c == None or self.c == "\0": break
                    if self.c == ".":
                        char += self.c
                        self.next()
                        if self.c == None or self.c == "\0": break

                        while self.c.isalnum():
                            char += self.c
                            self.next()
                            if self.c == None or self.c == "\0": break

                    elif self.c == 'i':
                        #this number has word suffix
                        char += self.c
                        self.next()
                        if self.c == None or self.c == "\0": break

                        while self.c.isdigit():
                            char += self.c
                            self.next()
                            if self.c == None or self.c == "\0": break

                    tokens.append(TokenInfo(1, char, start=start, end=self.pos.copy()))

                elif self.c == '.':
                    char = self.c
                    self.next()
                    if self.c == None or self.c == "\0": break

                    if self.c.isdigit():

                        while self.c.isdigit():
                            char += self.c
                            self.next()
                            if self.c == None or self.c == "\0": break

                        # Fix places like these, cuz now its break the whole thing and skip everything so .2454 at last is never going to register as an token
                        # so instead of breaking check if self.c is None if not then let it continue otherwise this is a messs
                        # for now stick with this later but defineatly write better code to remvoe these repetitive None checks.
                        if self.c == None or self.c == "\0": break
                        if self.c in 'Ee':
                            char += self.c
                            self.next()
                            if self.c == None or self.c == "\0": break

                            if self.c in '+-':
                                char += self.c
                                self.next()
                                if self.c == None or self.c == "\0": break

                            while self.c.isdigit():
                                char += self.c
                                self.next()
                                if self.c == None or self.c == "\0": break

                        tokens.append(TokenInfo(1, char, start=start, end=self.pos.copy()))
                        
                    else:
                        tokens.append(TokenInfo(3, char, exact_type=OPERATORS[char],start=start, end=self.pos.copy()))

                else:
                    char = self.c
                    self.next()
                    if self.c == None or self.c == "\0": break

                    while self.c.isdigit():
                        char += self.c
                        self.next()
                        if self.c == None or self.c == "\0": break
                    
                    if self.c == '.':

                        char = self.c
                        self.next()
                        if self.c == None or self.c == "\0": break

                        if self.c.isdigit():

                            while self.c.isdigit():
                                char += self.c
                                self.next()
                                if self.c == None or self.c == "\0": break
                        
                            if self.c in 'Ee':
                                char += self.c
                                self.next()
                                if self.c == None or self.c == "\0": break

                                if self.c in '+-':
                                    char += self.c
                                    self.next()
                                    if self.c == None or self.c == "\0": break
                                    
                                while self.c.isdigit():
                                    char += self.c
                                    self.next()
                                    if self.c == None or self.c == "\0": break
                    
                    if self.c in 'Ee':
                        char += self.c
                        self.next()
                        if self.c == None or self.c == "\0": break

                        if self.c in '+-':
                            char += self.c
                            self.next()
                            if self.c == None or self.c == "\0": break

                        while self.c.isdigit():
                            char += self.c
                            self.next()
                            if self.c == None or self.c == "\0": break

                    tokens.append(TokenInfo(1, char, start=start, end=self.pos.copy()))

            elif (char := self.c+self.peek(1)) in OPERATORS.keys():
                start = self.pos.copy()
                self.next(); self.next()
                tokens.append(TokenInfo(3, char, exact_type=OPERATORS[char], start=start, end=self.pos.copy()))
            
            elif self.c in OPERATORS.keys():
                start = self.pos.copy()
                char = self.c
                self.next()
                tokens.append(TokenInfo(3, char, exact_type=OPERATORS[char], start=start, end=self.pos.copy()))

            elif self.c == '"':
                start = self.pos.copy()
                self.next()
                if self.c == None or self.c == "\0": break

                string = ""
                while self.c != '"':
                    string += self.c
                    self.next()
                    if self.c == None or self.c == "\0": break

                self.next()
                tokens.append(TokenInfo(5, string, start=start, end=self.pos.copy()))

            elif self.c == "'":
                start = self.pos.copy()
                self.next()
                if self.c == None or self.c == "\0": break

                string = ""
                while self.c != "'":
                    string += self.c
                    self.next()
                    if self.c == None or self.c == "\0": break

                self.next()
                tokens.append(TokenInfo(5, string, start=start, end=self.pos.copy()))

            else:
                c = self.c
                self.next()
                tokens.append(TokenInfo(6, c, start=self.pos.copy(), end=self.pos.copy()))
                
        tokens.append(TokenInfo(0, 'END OF FILE', start=self.pos.copy(), end=self.pos.copy()))
        return tokens
    


# tokens = Tokenizer("2x + func(\"pro\") - *6//3#emblem .23445").tokenize()
# repr_tokens = []
# for i in tokens:
#     repr_tokens.append(
#         [
#             i.start,
#             TOKEN_TYPES[i.type],
#             i.exact_type,
#             i.string,
#             i.end
#         ]
#     )
# x = tabulate.tabulate(
#             repr_tokens,
#             headers=['start', 'type', 'exact_type', 'string', 'end'],
#             tablefmt='rounded_grid',
#             colalign=['center', 'center', 'center', 'center', 'center']
#  )
# print(x)