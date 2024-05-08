ENDMARKER = 0
NAME = 1
NUMBER = 2
STRING = 3
NEWLINE = 4
INDENT = 5
DEDENT = 6
LPAR = 7
RPAR = 8
LSQB = 9
RSQB = 10
COLON = 11
COMMA = 12
SEMI = 13
PLUS = 14
MINUS = 15
STAR = 16
SLASH = 17
VBAR = 18
AMPER = 19
LESS = 20
GREATER = 21
EQUAL = 22
DOT = 23
PERCENT = 24
BACKQUOTE = 25
LBRACE = 26
RBRACE = 27
EQEQUAL = 28
NOTEQUAL = 29
LESSEQUAL = 30
GREATEREQUAL = 31
TILDE = 32
CIRCUMFLEX = 33
LEFTSHIFT = 34
RIGHTSHIFT = 35
DOUBLESTAR = 36
PLUSEQUAL = 37
MINEQUAL = 38
STAREQUAL = 39
SLASHEQUAL = 40
PERCENTEQUAL = 41
AMPEREQUAL = 42
VBAREQUAL = 43
CIRCUMFLEXEQUAL = 44
LEFTSHIFTEQUAL = 45
RIGHTSHIFTEQUAL = 46
DOUBLESTAREQUAL = 47
OP = 48
ERRORTOKEN = 49
N_TOKENS = 50

DELIMITERS = {
    ',': True,
    '(': True,
    ')': True,
    '[': True,
    ']': True,
    '{': True,
    '}': True,
    '"': True,
    '\'': True,
    ';': True,

    '+': True,
    '-': True,
    '*': True,
    '.*': True,
    '/': True,
    './': True,
    '%': True,
    '^': True,
    '.^': True,
    '~': True,
    '!': True,
    '&': True,
    '|': True,
    '^|': True,
    '=': True,
    ':': True,
    '?': True,

    '==': True,
    '!=': True,
    '<': True,
    '>': True,
    '<=': True,
    '>=': True,

    '<<': True,
    '>>': True,
    '>>>': True # rightLogShift
}

DELIMITER_NAMES = [
    COMMA,
    LPAR,
    RPAR,
    LSQB,
    RSQB,
    LBRACE,
    RBRACE,
    SEMI,
    PLUS,
    MINUS,
    STAR,
    SLASH,
    PERCENT,
    CIRCUMFLEX,
    TILDE,
    NOTEQUAL,
    AMPER,
    VBAR,
    EQUAL,
    COLON,
    55, # for things like question mark
    EQEQUAL,
    NOTEQUAL,
    LESS,
    GREATER,
    LESSEQUAL,
    GREATEREQUAL,
    LEFTSHIFT,
    RIGHTSHIFT,
]

TOKENTYPE = {
    'NULL': 0,
    'DELIMITER': 1,
    'NUMBER': 2,
    'SYMBOL': 3,
    'UNKNOWN': 4
}

