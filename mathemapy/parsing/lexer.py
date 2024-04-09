from tokenize import tokenize
from io import BytesIO
import unicodedata

def pytokenizer(source: str):
    return list(tokenize(BytesIO(source.encode('utf-8')).readline))

# TOKEN TYPES DECLARATION
NULL = 0
DELIMITER = 1
NUMBER = 2
SYMBOL = 3
UNKNOWN = 4
EOF = -1
# DELIMITERS DECLARACTION ( START WITH 100 - )
COMMA = 100
LPAREN = 101
RPAREN = 102
LSQB = 103
RSQB = 104
LBRACE = 105
RBRACE = 106
DQUOTE = 107
SQUOTE = 108
SEMI = 109

PLUS = 110
MINUS = 111
STAR = 112
#DOTSTAR = 113
FOWARDSLASH = 114
#DOTFOWARDSLASH = 115
PERCENT = 116
CARROT = 117
#DOTCARROT = 118
TILDE = 119
EXCLAMATION = 120
AMPER = 121
BAR = 122
#CARROTBAR = 123
EQUAL = 124
COLON = 125
QUESTION = 126

EQEQUAL = 127
NOTEQUAL = 128
LESS = 129
GREATER = 130
LESSEQUAL = 131
GREATEREQUAL = 132

LEFTSHIFT = 133
RIGHTSHIFT = 134

BACKSLASH = 135

DELIMITERS = {
    ',': COMMA,
    '(': LPAREN,
    ')': RPAREN,
    '[': LSQB,
    ']': RSQB,
    '{': LBRACE,
    '}': RBRACE,
    '"': DQUOTE,
    '\'': SQUOTE,
    ';': SEMI,
    '+': PLUS,
    '-': MINUS,
    '*': STAR,
    # '.*': DOTSTAR,
    '/': FOWARDSLASH,
    # './': DOTFOWARDSLASH,
    '%': PERCENT,
    '^': CARROT,
    # '.^': DOTCARROT,
    '~': TILDE,
    '!': EXCLAMATION,
    '&': AMPER,
    '|': BAR,
    # '^|': CARROTBAR,
    '=': EQUAL,
    ':': COLON,
    '?': QUESTION,
    '==': EQEQUAL,
    '!=': NOTEQUAL,
    '<': LESS,
    '>': GREATER,
    '<=': LESSEQUAL,
    '>=': GREATEREQUAL,
    '<<': LEFTSHIFT,
    '>>': RIGHTSHIFT,
    # '>>>': ':'
}

# NAMED DELIMITERS
NAMED_DELIMITERS = [
    'mod',
    'to',
    'in',
    'and',
    'or',
    'not',
    'xor',
    'if',
    'else',
    'elseif',
    'for',
    'while'
]

#CONSTANTS
CONSTANTS = {
    'true': True,
    'false': False,
    'null': None,
    'undefined': 'undefined'
}

# NUMERIC CONSTANTS
NUMBERIC_CONSTANTS = ['NaN', 'infinity']

#ESCAPE CHARACTERS
ESCAPE_CHARACTERS = {
    'b': '\b',
    'n': '\n',
    't': '\t',
    'f': '\f',
    'r': '\r',
    DQUOTE: '"',
    SQUOTE: "'",
    BACKSLASH: '\\',
    FOWARDSLASH: '/',
}

class Position:
    def __init__(self, idx, ln, col, fn, ftxt ):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt
    
    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1
        if current_char == '\n':
            self.ln += 1
            self.col = 0
    
    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)
    
    def __repr__(self) -> str:
        return f"{self.idx};{self.ln}-{self.col}"
    
class TokenInfo:
    def __init__(self, type, exact_type=None, string=None, start=None, end=None) -> None:
        self.type = type
        self.string = string
        self.exact_type = exact_type if exact_type else type

        if start:
            self.start = start.copy()
            self.end = start.copy()
            self.end.advance()

        if end:
            self.end = end.copy()

    def matches(self, type, string):
        return self.type == type and self.string == string
    
    def __repr__(self) -> str:
        return f"start:{self.start} end:{self.end}\t\t{self.type}\t\t'{self.string}'"

class Tokenizer:
    def __init__(self, fn, encoding, content) -> None:
        self.fn = fn
        self.encoding = encoding
        self.content = content
        self.pos = Position(-1, 0, -1, fn, content)
        self.c = None
        self.next()

    def next(self):
        self.pos.advance(self.c)
        self.c = self.content[self.pos.idx] if self.pos.idx < len(self.content) else None

    def peek(self, offset=0):
        return self.content[self.pos.idx+offset] if ((self.pos.idx+offset) < len(self.content)) else None

    def tokenize(self):
        tokens = []
        tokens.append(TokenInfo('ENCODING', string=self.encoding, start=Position(0, 0, 0, self.fn, self.content)))

        while self.c != None and self.c != '':
            self.skipWhitespace()

            # check for number
            if (self.c.isdigit() or (self.c == '.')):
                start = self.pos.copy()
                if self.peek(1) != None:
                    prefix = self.c + self.peek(1)
                else:
                    prefix = self.c
                # checking for binary, octal, hex
                if prefix in ['0b', '0o', '0x']:
                    self.next()
                    self.next()
                    while self.c.isalnum():
                        self.next()
                        if self.c == None or self.c == "": break
                        
                    
                    # check if the number has a radix point
                    if self.c == '.':
                        self.next()
                        # collecting the digits after the radix
                        while self.c.isalnum():
                            self.next()
                            if self.c == None or self.c == "": break
                            
                    elif self.c == 'i':
                        # this number has word size suffix
                        self.next()
                        # get the word size
                        while self.c.isdigit():
                            self.next()
                            if self.c == None or self.c == "": break
                            
                    tokens.append(TokenInfo(NUMBER, string=self.content[start.idx:self.pos.idx], start=start, end=self.pos))

                # get a number, can have a single dot
                elif self.c == ".":
                    self.next()
                    # this is just the dot delimiter
                    if not self.c.isdigit():
                        tokens.append(TokenInfo(DELIMITER, exact_type=DELIMITERS[self.c],  string=self.c, start=start, end=self.pos))                   
                    else:
                        # for numbers like .2
                        while self.c.isdigit():
                            self.next()
                            if self.c == None or self.c == "": break
                            
                        tokens.append(TokenInfo(NUMBER, string=self.content[start.idx:self.pos.idx], start=start, end=self.pos))
                else:
                    while self.c.isdigit():
                        self.next()
                        if self.c == None or self.c == "": break
                        
                    # handle decimal places
                    if self.c == ".":
                        self.next()
                        while self.c.isdigit():
                            self.next()
                            if self.c == None or self.c == "": break
            
                        # check for expoentail notaion like "2.3e-4", "1.23e50"
                        if self.c in "Ee":
                            # 1.23e50 and 1.23e+5 are the same thing so i guess we need to check that as well
                            if self.c in "-+":
                                self.next()
                                while self.c.isdigit():
                                    self.next()
                                    if self.c == None or self.c == "": break
                                    
                                tokens.append(TokenInfo(NUMBER, string=self.content[start.idx:self.pos.idx], start=start, end=self.pos))
                            # here this will do that.
                            else:
                                self.next()
                                while self.c.isdigit():
                                    self.next()
                                    if self.c == None or self.c == "": break
                                tokens.append(TokenInfo(NUMBER, string=self.content[start.idx:self.pos.idx], start=start, end=self.pos))
                        else:
                            # just regular deimcal number
                            tokens.append(TokenInfo(NUMBER, string=self.content[start.idx:self.pos.idx], start=start, end=self.pos))
                    
                    #Exponential notation like "2e+4"
                    elif self.c == "e" or self.c == 'E':
                        # yeah its the same thing as up there,
                        if self.c in "-+":
                            self.next()
                            while self.c.isdigit():
                                self.next()
                                if self.c == None or self.c == "": break
                            tokens.append(TokenInfo(NUMBER, string=self.content[start.idx:self.pos.idx], start=start, end=self.pos))
                        else:
                            self.next()
                            while self.c.isdigit():
                                self.next()
                                if self.c == None or self.c == "": break
                            tokens.append(TokenInfo(NUMBER, string=self.content[start.idx:self.pos.idx], start=start, end=self.pos))
                    else:
                        # just regular whole number/ integer
                        tokens.append(TokenInfo(NUMBER, string=self.content[start.idx:self.pos.idx], start=start, end=self.pos))
                        
            # this must be here because of the dot handling
            elif self.c in DELIMITERS:
                delimiter = self.c
                self.next()
                # delimiters containing 2 characters
                if self.c is not None and (delimiter+self.c) in DELIMITERS:
                    delimiter += self.c
                    self.next()
                    # delimeters containting 3 charcters
                    if self.c is not None and (delimiter+self.c) in DELIMITERS:
                        delimiter += self.c
                        self.next()
                tokens.append(TokenInfo(DELIMITER, exact_type=DELIMITERS[delimiter], string=delimiter, start=start, end=self.pos))

            # check for variables, fucntion names and named operators
            elif self.c == None or self.c == "": break
            elif self.isvalidmath():
                start = self.pos.copy()
                self.next()
                if self.c == None or self.c == "": break
                while self.isvalidmath():
                    if self.c == None or self.c == "": break
                    self.next()
                
                if self.content[start.idx:self.pos.idx] in NAMED_DELIMITERS:
                    tokens.append(TokenInfo(DELIMITER, exact_type=NAMED_DELIMITERS[self.content[start.idx:self.pos.idx]], string=self.content[start.idx:self.pos.idx], start=start, end=self.pos))

                else:
                    tokens.append(TokenInfo(SYMBOL, string=self.content[start.idx:self.pos.idx], start=start, end=self.pos))
        
        tokens.append(TokenInfo(EOF, string='EOF', start=self.pos))
        return tokens
    
    def skipWhitespace(self):
        while self.c != None and self.c.isspace():
            self.next()

    def isvalidmath(self):
        if self.c in "_$": return True
        elif self.c.isalnum(): return True
        elif unicodedata.name(self.c).startswith('GREEK'): return True
        elif unicodedata.name(self.c).startswith('LATIN'): return True
        else: return False

def run_tests():
    fn = "<stdin>"
    contents = "2x+5-(9*7)-5/4 + 23e5"
    tokens = Tokenizer(fn=fn, encoding='utf-8', content=contents).tokenize()
    return tokens