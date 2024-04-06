from abc import ABC, abstractmethod
from fractions import Fraction
from decimal import Decimal

"""

THIS IS A DIRECT IMPLEMENTATION OF THE MATHJS PARSER.
NOTE: THERE ARE SO MANY NOTABLE BUGS IN HERE.
IMPLEMENT PURELY FOR THE PURPOSE OF UNDERSTANDING THE MATHJS PARSER.
PLEASE TAKE A VISIT TO MATHJS FOR MORE INFORMATION: https://github.com/josdejong/mathjs

"""
# Define valid input types
valid_input_types = {
    'string': True,
    'number': True,
    'Decimal': True,
    'Fraction': True
}
DEFAULT_CONFIG = {
    # minimum relative difference between two compared values,
    # used by all comparison functions
    'epsilon': 1e-12,

    # type of default matrix output. Choose 'matrix' (default) or 'array'
    'matrix': 'Matrix',

    # type of default number output. Choose 'number' (default), 'BigNumber', or 'Fraction'
    'number': 'number',

    # number of significant digits in BigNumbers
    'precision': 64,

    # predictable output type of functions. When true, output type depends only
    # on the input types. When false (default), output type can vary depending
    # on input values. For example `math.sqrt(-4)` returns `complex('2i')` when
    # predictable is false, and returns `NaN` when true.
    'predictable': False,

    # random seed for seeded pseudo random number generation
    # None = randomly seed
    'randomSeed': None
}
MATRIX_OPTIONS = ['Matrix', 'Array']  # valid values for option matrix
NUMBER_OPTIONS = ['number', 'BigNumber', 'Fraction']  # valid values for option number


def config(options=None):
    if options:
        raise ValueError('The global config is readonly. Please create a math module instance if you want to change the default configuration.')

    return DEFAULT_CONFIG.copy()

config.MATRIX_OPTIONS = MATRIX_OPTIONS
config.NUMBER_OPTIONS = NUMBER_OPTIONS
config.number = DEFAULT_CONFIG['number']
# Define functions for converting to each output type
def number(x):
    return float(x)

def bignumber(x):
    return Decimal(x)

def fraction(x):
    return Fraction(x)

def no_bignumber(x):
    raise NotImplementedError("BigNumber conversion not supported")

def no_fraction(x):
    raise NotImplementedError("Fraction conversion not supported")

valid_output_types = {
    'number': number,
    'Decimal': bignumber,
    'Fraction': fraction
}

# Define numeric conversion function
def numeric(value, output_type='number', check=None):
    if check is not None:
        raise SyntaxError('numeric() takes one or two arguments')
    input_type = type(value).__name__

    if input_type not in valid_input_types:
        pass#raise TypeError('Cannot convert {} of type "{}"; valid input types are {}'.format(value, input_type, ', '.join(valid_input_types.keys())))

    if output_type not in valid_output_types:
        pass#raise TypeError('Cannot convert {} to type "{}"; valid output types are {}'.format(value, output_type, ', '.join(valid_output_types.keys())))

    if output_type == input_type:
        return value
    else:
        return valid_output_types[output_type](value)


# NODES

# class Node:
#     @property
#     def type(self):
#         return 'Node'

#     @property
#     def is_node(self):
#         return True

def is_accessor_node(x):
    return getattr(x, 'isAccessorNode', False) and getattr(x.__class__.__bases__[0], 'isNode', False)

def is_constant_node(x):
    return getattr(x, 'isConstantNode', False) and getattr(x.__class__.__bases__[0], 'isNode', False)

def is_function_node(x):
    return getattr(x, 'isFunctionNode', False) and getattr(x.__class__.__bases__[0], 'isNode', False)


def is_operator_node(x):
    return getattr(x, 'isOperatorNode', False) and getattr(x.__class__.__bases__[0], 'isNode', False)


def is_symbol_node(x):
    return getattr(x, 'isSymbolNode', False) and getattr(x.__class__.__bases__[0], 'isNode', False)

def rule2_node(node):
    return is_constant_node(node) or (
        is_operator_node(node) and
        len(node.args) == 1 and
        is_constant_node(node.args[0]) and
        node.op in '-+~'
    )

def parse_string(expression):
    return parse_start(expression, {})

def parse_array_or_matrix(expressions):
    return parse_multiple(expressions, {})

def parse_string_with_object(expression, options):
    extra_nodes = options.get('nodes', {})
    return parse_start(expression, extra_nodes)

def parse_multiple_with_object(expressions, options):
    return parse_multiple(expressions, options)

parse = {
    str: parse_string,
    list: parse_array_or_matrix,
    tuple: parse_array_or_matrix,
    (str, dict): parse_string_with_object,
    (list, dict): parse_multiple_with_object
}

def parse_multiple(expressions, options={}):
    extra_nodes = options.get('nodes', {})
    
    # Define a recursive function to handle nested structures
    def recursive_parse(elem):
        if not isinstance(elem, str):
            raise TypeError('String expected')
        
        return parse_start(elem, extra_nodes)
    
    # Apply the recursive_parse function to each element in the expressions
    return [recursive_parse(elem) for elem in expressions]


class Node(ABC):
    @property
    @abstractmethod
    def type(self):
        pass

    @property
    def is_node(self):
        return True

class IndexNode(Node):
    def __init__(self, dimensions, dot_notation=False):
        super().__init__()
        self.dimensions = dimensions
        self.dot_notation = dot_notation

        # Validate input
        if not isinstance(dimensions, list) or not all(isinstance(dim, Node) for dim in dimensions):
            raise TypeError('List containing Nodes expected for parameter "dimensions"')
        if self.dot_notation and not self.is_object_property():
            raise ValueError('dotNotation only applicable for object properties')

    @property
    def type(self):
        return self.name

    @property
    def is_index_node(self):
        return True

    def __repr__(self) -> str:
        return str(
            {
                "dimensions": self.dimensions,
                "dotNotation": self.dot_notation
            }
        )

class AccessorNode(Node):
    def __init__(self, object, index):
        if not isinstance(object, Node):
            raise TypeError('Node expected for parameter "object"')
        if not isinstance(index, IndexNode):
            raise TypeError('IndexNode expected for parameter "index"')

        self.object = object
        self.index = index

    @property
    def name(self):
        if self.index:
            return self.index.get_object_property() if self.index.is_object_property() else ''
        else:
            return self.object.name if hasattr(self.object, 'name') else ''

    @property
    def type(self):
        return self.name

    @property
    def is_accessor_node(self):
        return True
    
    def __repr__(self) -> str:
        return str(
            {
                "object": self.object,
                "index": self.index
            }
        )


class ArrayNode(Node):
    def __init__(self, items=None):
        super().__init__()
        self.items = items or []

        # Validate input
        if not isinstance(self.items, list) or not all(isinstance(item, Node) for item in self.items):
            raise TypeError('List containing Nodes expected')

    @property
    def type(self):
        return self.name

    @property
    def is_array_node(self):
        return True
    
    def __repr__(self) -> str:
        return str(
            {
                "items": self.items
            }
        )
    
class AssignmentNode(Node):
    """
    Define a symbol, like `a=3.2`, update a property like `a.b=3.2`, or replace a subset of a matrix like `A[2,2]=42`.
    """
    def __init__(self, object, index=None, value=None):
        super().__init__()
        self.object = object
        self.index = index
        self.value = value or index

        # Validate input
        if not isinstance(object, (SymbolNode, AccessorNode)):
            raise TypeError('SymbolNode or AccessorNode expected as "object"')
        if isinstance(object, SymbolNode) and object.name == 'end':
            raise ValueError('Cannot assign to symbol "end"')
        if self.index and not isinstance(self.index, IndexNode):
            raise TypeError('IndexNode expected as "index"')
        if not isinstance(self.value, Node):
            raise TypeError('Node expected as "value"')

    @property
    def name(self):
        if self.index:
            return self.index.object_property if self.index.is_object_property else ''
        else:
            return self.object.name or ''

    @property
    def type(self):
        return name

    @property
    def is_assignment_node(self):
        return True
    
    def __repr__(self) -> str:
        return str(
            {
                "object": self.object,
                "index": self.index,
                "value": self.value
            }
        )

class BlockNode(Node):
    """
    Holds a set with blocks
    """
    def __init__(self, blocks):
        super().__init__()
        # Validate input, copy blocks
        if not isinstance(blocks, list):
            raise TypeError('List expected')
        self.blocks = []
        for block in blocks:
            node = block.get('node')
            visible = block.get('visible', True)

            if not isinstance(node, Node):
                raise TypeError('Property "node" must be a Node')
            if not isinstance(visible, bool):
                raise TypeError('Property "visible" must be a boolean')

            self.blocks.append({'node': node, 'visible': visible})

    @property
    def type(self):
        return name

    @property
    def is_block_node(self):
        return True
    
    def __repr__(self) -> str:
        return str(
            {
                "blocks": self.blocks
            }
        )
    

class ConditionalNode(Node):
    """
    A lazy evaluating conditional operator: 'condition ? trueExpr : falseExpr'
    """
    def __init__(self, condition, true_expr, false_expr):
        super().__init__()
        if not isinstance(condition, Node):
            raise TypeError('Parameter condition must be a Node')
        if not isinstance(true_expr, Node):
            raise TypeError('Parameter trueExpr must be a Node')
        if not isinstance(false_expr, Node):
            raise TypeError('Parameter falseExpr must be a Node')

        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr

    @property
    def type(self):
        return name

    @property
    def is_conditional_node(self):
        return True
    
    def __repr__(self) -> str:
        return str(
            {
                "condition": self.condition,
                "trueExpr": self.true_expr,
                "falseExpr": self.false_expr
            }
        )

class ConstantNode(Node):
    """
    A ConstantNode holds a constant value like a number or string.
    """
    def __init__(self, value):
        super().__init__()
        self.value = value

    @property
    def type(self):
        return name

    @property
    def is_constant_node(self):
        return True
    
    def is_percentage(self):
        return False
    
    def __repr__(self) -> str:
        return str(
            {
                "value": self.value
            }
        )

keywords = {'end'}

class FunctionAssignmentNode(Node):
    """
    Function assignment
    """

    def __init__(self, name, params, expr):
        super().__init__()
        # Validate input
        if not isinstance(name, str):
            raise TypeError('String expected for parameter "name"')
        if not isinstance(params, list):
            raise TypeError('List expected for parameter "params"')
        if not isinstance(expr, Node):
            raise TypeError('Node expected for parameter "expr"')
        if name in keywords:
            raise ValueError('Illegal function name, "' + name + '" is a reserved keyword')

        param_names = set()
        self.params = []
        self.types = []
        for param in params:
            if isinstance(param, str):
                param_name = param
                param_type = 'any'
            elif isinstance(param, dict) and 'name' in param:
                param_name = param['name']
                param_type = param.get('type', 'any')
            else:
                raise TypeError('Invalid parameter format')
            
            if param_name in param_names:
                raise ValueError(f'Duplicate parameter name "{param_name}"')
            else:
                param_names.add(param_name)
            
            self.params.append(param_name)
            self.types.append(param_type)

        self.name = name
        self.expr = expr

    @property
    def type(self):
        return name

    @property
    def is_function_assignment_node(self):
        return True
    
    def __repr__(self) -> str:
        return str(
            {
                "name": self.name,
                "params": self.params,
                "expr": self.expr
            }
        )

class FunctionNode(Node):
    """
    Invoke a list with arguments on a node
    """

    def __init__(self, fn, args):
        super().__init__()

        if isinstance(fn, str):
            fn = SymbolNode(fn)

        # Validate input
        if not isinstance(fn, Node):
            raise TypeError('Node expected as parameter "fn"')
        if not isinstance(args, list) or not all(isinstance(arg, Node) for arg in args):
            raise TypeError('List of nodes expected for parameter "args"')

        self.fn = fn
        self.args = args

    @property
    def name(self):
        return self.fn.name if hasattr(self.fn, 'name') else ''

    @property
    def type(self):
        return 'FunctionNode'

    @property
    def is_function_node(self):
        return True
    
    def __repr__(self) -> str:
        return str(
            {
                "fn": self.fn,
                "args": self.args
            }
        )
    

class ObjectNode(Node):
    """
    Holds an object with keys/values
    """

    def __init__(self, properties=None):
        super().__init__()
        self.properties = properties or {}

        # Validate input
        if properties:
            if not isinstance(properties, dict) or not all(isinstance(value, Node) for value in properties.values()):
                raise TypeError('Dictionary containing Nodes expected')

    @property
    def type(self):
        return 'ObjectNode'

    @property
    def is_object_node(self):
        return True
    
    def __repr__(self) -> str:
        return str(
            {
                "properties": self.properties
            }
        )
    

class OperatorNode(Node):
    """
    An operator with two arguments, like 2+3
    """

    def __init__(self, op, fn, args, implicit=False, is_percentage=False):
        super().__init__()
        # Validate input
        if not isinstance(op, str):
            raise TypeError('String expected for parameter "op"')
        if not isinstance(fn, str):
            raise TypeError('String expected for parameter "fn"')
        if not isinstance(args, list) or not all(isinstance(arg, Node) for arg in args):
            raise TypeError('List containing Nodes expected for parameter "args"')

        self.implicit = implicit
        self.is_percentage = is_percentage
        self.op = op
        self.fn = fn
        self.args = args or []

    @property
    def type(self):
        return 'OperatorNode'

    @property
    def is_operator_node(self):
        return True
    
    def __repr__(self) -> str:
        return str(
            {
                "op": self.op,
                "fn": self.fn,
                "args": self.args,
                "implicit": self.implicit,
                "isPercentage": self.is_percentage
            }
        )
class ParenthesisNode(Node):
    """
    A parenthesis node describes manual parenthesis from the user input
    """

    def __init__(self, content):
        super().__init__()
        # Validate input
        if not isinstance(content, Node):
            raise TypeError('Node expected for parameter "content"')

        self.content = content

    @property
    def type(self):
        return 'ParenthesisNode'

    @property
    def is_parenthesis_node(self):
        return True
    
    def __repr__(self) -> str:
        return str(
            {
                "content": self.content
            }
        )

class RangeNode(Node):
    """
    Create a range
    """

    def __init__(self, start, end, step=None):
        super().__init__()
        # Validate inputs
        if not isinstance(start, Node):
            raise TypeError('Node expected for start')
        if not isinstance(end, Node):
            raise TypeError('Node expected for end')
        if step is not None and not isinstance(step, Node):
            raise TypeError('Node expected for step')
        if len(locals()) > 4:
            raise ValueError('Too many arguments')

        self.start = start  # Included lower-bound
        self.end = end  # Included upper-bound
        self.step = step  # Optional step

    @property
    def type(self):
        return 'RangeNode'

    @property
    def is_range_node(self):
        return True
    
    def __repr__(self) -> str:
        return str(
            {
                "start": self.start,
                "end": self.end,
                "step": self.step
            }
        )

class RelationalNode(Node):
    """
    A node representing a chained conditional expression, such as 'x > y > z'
    """

    def __init__(self, conditionals, params):
        super().__init__()
        if not isinstance(conditionals, list):
            raise TypeError('Parameter conditionals must be a list')
        if not isinstance(params, list):
            raise TypeError('Parameter params must be a list')
        if len(conditionals) != len(params) - 1:
            raise TypeError('Parameter params must contain exactly one more element than parameter conditionals')

        self.conditionals = conditionals
        self.params = params

    @property
    def type(self):
        return 'RelationalNode'

    @property
    def is_relational_node(self):
        return True
    
    def __repr__(self) -> str:
        return str(
            {
                "conditionals": self.conditionals,
                "params": self.params
            }
        )

class SymbolNode(Node):
    """
    A symbol node can hold and resolve a symbol
    """

    def __init__(self, name):
        super().__init__()
        if not isinstance(name, str):
            raise TypeError('String expected for parameter "name"')

        self.name = name

    @property
    def type(self):
        return 'SymbolNode'

    @property
    def is_symbol_node(self):
        return True
    
    def __repr__(self) -> str:
        return str(
            {
                "name": self.name
            }
        )



TOKENTYPE = {
    'NULL': 0,
    'DELIMITER': 1,
    'NUMBER': 2,
    'SYMBOL': 3,
    'UNKNOWN': 4
}

# map with all delimiters
DELIMITERS = {
    ',': True,
    '(': True,
    ')': True,
    '[': True,
    ']': True,
    '{': True,
    '}': True,
    '"': True,
    "'": True,
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
    '>>>': True
}

# map with all named delimiters
NAMED_DELIMITERS = {
    'mod': True,
    'to': True,
    'in': True,
    'and': True,
    'xor': True,
    'or': True,
    'not': True
}

CONSTANTS = {
    'true': True,
    'false': False,
    'null': None,
    'undefined': None
}

NUMERIC_CONSTANTS = [
    'NaN',
    'Infinity'
]

ESCAPE_CHARACTERS = {
    '"': '"',
    "'": "'",
    '\\': '\\',
    '/': '/',
    'b': '\b',
    'f': '\f',
    'n': '\n',
    'r': '\r',
    't': '\t'
    # note that \u is handled separately in parseStringToken()
}

def initial_state():
    return {
        'extraNodes': {},  # current extra nodes, must be careful not to mutate
        'expression': '',  # current expression
        'comment': '',  # last parsed comment
        'index': 0,  # current index in expr
        'token': '',  # current token
        'tokenType': TOKENTYPE['NULL'],  # type of the token
        'nestingLevel': 0,  # level of nesting inside parameters, used to ignore newline characters
        'conditionalLevel': None  # when a conditional is being parsed, the level of the conditional is stored here
    }

def current_string(state, length=1):
    return state['expression'][state['index']:state['index']+length]

def current_character(state):
    return current_string(state, 1)

def next(state):
    state['index'] += 1

def prev_character(state):
    return state['expression'][state['index'] - 1]

def next_character(state):
    try:
        return state['expression'][state['index'] + 1]
    except IndexError:
        return None


def get_token(state):
    state["tokenType"] = TOKENTYPE['NULL']
    state["token"] = ""
    state["comment"] = ""

    # skip over ignored characters
    while True:
        # comments
        if current_character(state) == '#':
            while current_character(state) != '\n' and current_character(state) != '':
                state["comment"] += current_character(state)
                next(state)
        # whitespace: space, tab, and newline when inside parameters
        if is_whitespace(current_character(state), state["nestingLevel"]):
            next(state)
        else:
            break

    # check for end of expression
    if current_character(state) == '':
        # token is still empty
        state["tokenType"] = TOKENTYPE['DELIMITER']
        return

    # check for new line character
    if current_character(state) == '\n' and not state["nestingLevel"]:
        state["tokenType"] = TOKENTYPE['DELIMITER']
        state["token"] = current_character(state)
        next(state)
        return

    c1 = current_character(state)
    c2 = current_string(state, 2)
    c3 = current_string(state, 3)
    if len(c3) == 3 and DELIMITERS.get(c3):
        state["tokenType"] = TOKENTYPE['DELIMITER']
        state["token"] = c3
        next(state)
        next(state)
        next(state)
        return

    # check for delimiters consisting of 2 characters
    if len(c2) == 2 and DELIMITERS.get(c2):
        state["tokenType"] = TOKENTYPE['DELIMITER']
        state["token"] = c2
        next(state)
        next(state)
        return

    # check for delimiters consisting of 1 character
    if DELIMITERS.get(c1):
        state["tokenType"] = TOKENTYPE['DELIMITER']
        state["token"] = c1
        next(state)
        return

    # check for a number
    if is_digit_dot(c1):
        state["tokenType"] = TOKENTYPE['NUMBER']

        # check for binary, octal, or hex
        c2 = current_string(state, 2)
        if c2 == '0b' or c2 == '0o' or c2 == '0x':
            state["token"] += current_character(state)
            next(state)
            state["token"] += current_character(state)
            next(state)
            while is_hex_digit(current_character(state)):
                state["token"] += current_character(state)
                next(state)
            if current_character(state) == '.':
                # this number has a radix point
                state["token"] += '.'
                next(state)
                # get the digits after the radix
                while is_hex_digit(current_character(state)):
                    state["token"] += current_character(state)
                    next(state)
            elif current_character(state) == 'i':
                # this number has a word size suffix
                state["token"] += 'i'
                next(state)
                # get the word size
                while is_digit(current_character(state)):
                    state["token"] += current_character(state)
                    next(state)
            return

        # get number, can have a single dot
        if current_character(state) == '.':
            state["token"] += current_character(state)
            next(state)

            if not is_digit(current_character(state)):
                # this is no number, it is just a dot (can be dot notation)
                state["tokenType"] = TOKENTYPE['DELIMITER']
                return
        else:
            while is_digit(current_character(state)):
                state["token"] += current_character(state)
                next(state)
            if is_decimal_mark(current_character(state), next_character(state)):
                state["token"] += current_character(state)
                next(state)

        while is_digit(current_character(state)):
            state["token"] += current_character(state)
            next(state)
        # check for exponential notation like "2.3e-4", "1.23e50" or "2e+4"
        if current_character(state) == 'E' or current_character(state) == 'e':
            if is_digit(next_character(state)) or next_character(state) == '-' or next_character(state) == '+':
                state["token"] += current_character(state)
                next(state)

                if current_character(state) == '+' or current_character(state) == '-':
                    state["token"] += current_character(state)
                    next(state)
                # Scientific notation MUST be followed by an exponent
                if not is_digit(current_character(state)):
                    raise create_syntax_error(state, 'Digit expected, got "' + current_character(state) + '"')

                while is_digit(current_character(state)):
                    state["token"] += current_character(state)
                    next(state)

                if is_decimal_mark(current_character(state), next_character(state)):
                    raise create_syntax_error(state, 'Digit expected, got "' + current_character(state) + '"')
            elif next_character(state) == '.':
                next(state)
                raise create_syntax_error(state, 'Digit expected, got "' + current_character(state) + '"')

        return

    # check for variables, functions, named operators
    if is_alpha(current_character(state), prev_character(state), next_character(state)):
        while is_alpha(current_character(state), prev_character(state), next_character(state)) or is_digit(current_character(state)):
            state["token"] += current_character(state)
            next(state)

        if state["token"] in NAMED_DELIMITERS:
            state["tokenType"] = TOKENTYPE['DELIMITER']
        else:
            state["tokenType"] = TOKENTYPE['SYMBOL']

        return

    # something unknown is found, wrong characters -> a syntax error
    state["tokenType"] = TOKENTYPE['UNKNOWN']
    while current_character(state) != '':
        state["token"] += current_character(state)
        next(state)
    raise create_syntax_error(state, 'Syntax error in part "' + state["token"] + '"')

def get_token_skip_newline(state):
    while True:
        get_token(state)
        if state["token"] != '\n':
            break

def open_params(state):
    state["nestingLevel"] += 1

def close_params(state):
    state["nestingLevel"] -= 1

def is_alpha(c, c_prev, c_next):
    return is_valid_latin_or_greek(c) or is_valid_math_symbol(c, c_next) or is_valid_math_symbol(c_prev, c)

def is_valid_latin_or_greek(c):
    return c.isalpha() or c in ['_', '$'] or '\u00C0' <= c <= '\u02AF' or '\u0370' <= c <= '\u03FF' or '\u2100' <= c <= '\u214F'

def is_valid_math_symbol(high, low):
    return high == '\uD835' and '\uDC00' <= low <= '\uDFFF' and low not in ['\uDC55', '\uDC9D', '\uDCA0', '\uDCA1', '\uDCA3', '\uDCA4', '\uDCA7', '\uDCA8', '\uDCAD', '\uDCBA', '\uDCBC', '\uDCC4', '\uDD06', '\uDD0B', '\uDD0C', '\uDD15', '\uDD1D', '\uDD3A', '\uDD3F', '\uDD45', '\uDD47', '\uDD48', '\uDD49', '\uDD51', '\uDEA6', '\uDEA7', '\uDFCC', '\uDFCD']

def is_whitespace(c, nesting_level):
    return c in [' ', '\t'] or (c == '\n' and nesting_level > 0)

def is_decimal_mark(c, c_next):
    return c == '.' and c_next not in ['/', '*', '^']

def is_digit_dot(c):
    return c.isdigit() or c == '.'

def is_digit(c):
    return c.isdigit()

def is_hex_digit(c):
    return c.isdigit() or ('a' <= c <= 'f') or ('A' <= c <= 'F')

def parse_start(expression, extra_nodes):
    state = initial_state()
    state.update({"expression": expression, "extraNodes": extra_nodes})
    get_token(state)

    node = parse_block(state)

    # check for garbage at the end of the expression
    # an expression ends with an empty character '' and tokenType DELIMITER
    if state["token"] != '':
        if state["tokenType"] == TOKENTYPE['DELIMITER']:
            # user entered a non-existing operator like "//"
            # TODO: give hints for aliases, for example with "<>" give as hint "did you mean !== ?"
            raise create_error(state, 'Unexpected operator ' + state["token"])
        else:
            raise create_syntax_error(state, 'Unexpected part "' + state["token"] + '"')

    return node

def parse_block(state):
    node = None
    blocks = []
    visible = None

    if state["token"] != '' and state["token"] != '\n' and state["token"] != ';':
        node = parse_assignment(state)
        if state["comment"]:
            node.comment = state["comment"]

    # TODO: simplify this loop
    while state["token"] == '\n' or state["token"] == ';':
        if len(blocks) == 0 and node:
            visible = (state["token"] != ';')
            blocks.append({"node": node, "visible": visible})

        get_token(state)
        if state["token"] != '\n' and state["token"] != ';' and state["token"] != '':
            node = parse_assignment(state)
            if state["comment"]:
                node.comment = state["comment"]

            visible = (state["token"] != ';')
            blocks.append({"node": node, "visible": visible})

    if len(blocks) > 0:
        return BlockNode(blocks)
    else:
        if not node:
            node = ConstantNode(None)
            if state["comment"]:
                node.comment = state["comment"]

        return node

def parse_assignment(state):
    name = None
    args = None
    value = None
    valid = None

    node = parse_conditional(state)

    if state["token"] == '=':
        if is_symbol_node(node):
            # parse a variable assignment like 'a = 2/3'
            name = node.name
            get_token_skip_newline(state)
            value = parse_assignment(state)
            return AssignmentNode(SymbolNode(name), value)
        elif is_accessor_node(node):
            # parse a matrix subset assignment like 'A[1,2] = 4'
            get_token_skip_newline(state)
            value = parse_assignment(state)
            return AssignmentNode(node.object, node.index, value)
        elif is_function_node(node) and is_symbol_node(node.fn):
            # parse function assignment like 'f(x) = x^2'
            valid = True
            args = []

            name = node.name
            for arg in node.args:
                if is_symbol_node(arg):
                    args.append(arg.name)
                else:
                    valid = False

            if valid:
                get_token_skip_newline(state)
                value = parse_assignment(state)
                return FunctionAssignmentNode(name, args, value)

        raise create_syntax_error(state, 'Invalid left hand side of assignment operator =')

    return node

def parse_conditional(state):
    node = parse_logical_or(state)

    while state["token"] == '?':
        # set a conditional level, the range operator will be ignored as long
        # as conditionalLevel === state.nestingLevel.
        prev = state["conditionalLevel"]
        state["conditionalLevel"] = state["nestingLevel"]
        get_token_skip_newline(state)

        condition = node
        true_expr = parse_assignment(state)

        if state["token"] != ':':
            raise create_syntax_error(state, 'False part of conditional expression expected')

        state["conditionalLevel"] = None
        get_token_skip_newline(state)

        false_expr = parse_assignment(state)  # Note: check for conditional operator again, right associativity

        node = ConditionalNode(condition, true_expr, false_expr)

        # restore the previous conditional level
        state["conditionalLevel"] = prev

    return node

def parse_logical_or(state):
    node = parse_logical_xor(state)

    while state["token"] == 'or':
        get_token_skip_newline(state)
        node = OperatorNode('or', 'or', [node, parse_logical_xor(state)])

    return node


def parse_logical_xor(state):
    node = parse_logical_and(state)

    while state["token"] == 'xor':
        get_token_skip_newline(state)
        node = OperatorNode('xor', 'xor', [node, parse_logical_and(state)])

    return node


def parse_logical_and(state):
    node = parse_bitwise_or(state)

    while state["token"] == 'and':
        get_token_skip_newline(state)
        node = OperatorNode('and', 'and', [node, parse_bitwise_or(state)])

    return node


def parse_bitwise_or(state):
    node = parse_bitwise_xor(state)

    while state["token"] == '|':
        get_token_skip_newline(state)
        node = OperatorNode('|', 'bitOr', [node, parse_bitwise_xor(state)])

    return node


def parse_bitwise_xor(state):
    node = parse_bitwise_and(state)

    while state["token"] == '^|':
        get_token_skip_newline(state)
        node = OperatorNode('^|', 'bitXor', [node, parse_bitwise_and(state)])

    return node

def parse_bitwise_and(state):
    node = parse_relational(state)

    while state["token"] == '&':
        get_token_skip_newline(state)
        node = OperatorNode('&', 'bitAnd', [node, parse_relational(state)])

    return node


def parse_relational(state):
    params = [parse_shift(state)]
    conditionals = []

    operators = {
        '==': 'equal',
        '!=': 'unequal',
        '<': 'smaller',
        '>': 'larger',
        '<=': 'smallerEq',
        '>=': 'largerEq'
    }

    while state["token"] in operators:
        cond = {"name": state["token"], "fn": operators[state["token"]]}
        conditionals.append(cond)
        get_token_skip_newline(state)
        params.append(parse_shift(state))

    if len(params) == 1:
        return params[0]
    elif len(params) == 2:
        return OperatorNode(conditionals[0]["name"], conditionals[0]["fn"], params)
    else:
        return RelationalNode([c["fn"] for c in conditionals], params)


def parse_shift(state):
    node = parse_conversion(state)
    operators = {
        '<<': 'leftShift',
        '>>': 'rightArithShift',
        '>>>': 'rightLogShift'
    }

    while state["token"] in operators:
        name = state["token"]
        fn = operators[name]
        get_token_skip_newline(state)
        params = [node, parse_conversion(state)]
        node = OperatorNode(name, fn, params)

    return node

def parse_conversion(state):
    node = parse_range(state)
    operators = {
        'to': 'to',
        'in': 'to'  # alias of 'to'
    }

    while state["token"] in operators:
        name = state["token"]
        fn = operators[name]
        get_token_skip_newline(state)

        if name == 'in' and state["token"] == '':
            # end of expression -> this is the unit 'in' ('inch')
            node = OperatorNode('*', 'multiply', [node, SymbolNode('in')], True)
        else:
            # operator 'a to b' or 'a in b'
            params = [node, parse_range(state)]
            node = OperatorNode(name, fn, params)

    return node


def parse_range(state):
    if state["token"] == ':':
        # implicit start=1 (one-based)
        node = ConstantNode(1)
    else:
        # explicit start
        node = parse_add_subtract(state)

    params = []
    if state["token"] == ':' and (state["conditionalLevel"] != state["nestingLevel"]):
        # we ignore the range operator when a conditional operator is being processed on the same level
        params.append(node)

        # parse step and end
        while state["token"] == ':' and len(params) < 3:
            get_token_skip_newline(state)

            if state["token"] in [')', ']', ',', '']:
                # implicit end
                params.append(SymbolNode('end'))
            else:
                # explicit end
                params.append(parse_add_subtract(state))

        if len(params) == 3:
            # params = [start, step, end]
            node = RangeNode(params[0], params[2], params[1])  # start, end, step
        else:
            # params = [start, end]
            node = RangeNode(params[0], params[1])  # start, end

    return node


def parse_add_subtract(state):
    node = parse_multiply_divide(state)
    operators = {
        '+': 'add',
        '-': 'subtract'
    }
    while state["token"] in operators:
        name = state["token"]
        fn = operators[name]
        get_token_skip_newline(state)
        right_node = parse_multiply_divide(state)
        params = [node, right_node] if not right_node.is_percentage else [node, OperatorNode('*', 'multiply', [node, right_node])]
        node = OperatorNode(name, fn, params)

    return node

def parse_multiply_divide(state):
    node = parse_implicit_multiplication(state)
    last = node
    operators = {
        '*': 'multiply',
        '.*': 'dotMultiply',
        '/': 'divide',
        './': 'dotDivide'
    }

    while True:
        if state["token"] in operators:
            # explicit operators
            name = state["token"]
            fn = operators[name]
            get_token_skip_newline(state)
            last = parse_implicit_multiplication(state)
            node = OperatorNode(name, fn, [node, last])
        else:
            break

    return node


def parse_implicit_multiplication(state):
    node = parse_rule2(state)
    last = node

    while True:
        if (state["tokenType"] == TOKENTYPE['SYMBOL'] or
                (state["token"] == 'in' and is_constant_node(node)) or
                (state["tokenType"] == TOKENTYPE['NUMBER'] and
                 not is_constant_node(last) and
                 (not is_operator_node(last) or last.op == '!')) or
                (state["token"] == '(')):
            # parse implicit multiplication
            last = parse_rule2(state)
            node = OperatorNode('*', 'multiply', [node, last], True)  # implicit
        else:
            break

    return node


def parse_rule2(state):
    node = parse_percentage(state)
    last = node
    token_states = []

    while True:
        # Match the "number /" part of the pattern "number / number symbol"
        if state["token"] == '/' and rule2_node(last):
            # Look ahead to see if the next token is a number
            token_states.append(state.copy())
            get_token_skip_newline(state)

            # Match the "number / number" part of the pattern
            if state["tokenType"] == TOKENTYPE['NUMBER']:
                # Look ahead again
                token_states.append(state.copy())
                get_token_skip_newline(state)

                # Match the "symbol" part of the pattern, or a left parenthesis
                if state["tokenType"] == TOKENTYPE['SYMBOL'] or state["token"] == '(':
                    # We've matched the pattern "number / number symbol".
                    # Rewind once and build the "number / number" node; the symbol will be consumed later
                    state = token_states.pop()
                    token_states.pop()
                    last = parse_percentage(state)
                    node = OperatorNode('/', 'divide', [node, last])
                else:
                    # Not a match, so rewind
                    token_states.pop()
                    state = token_states.pop()
                    break
            else:
                # Not a match, so rewind
                state = token_states.pop()
                break
        else:
            break

    return node

def parse_percentage(state):
    node = parse_unary(state)
    operators = {'%': 'mod', 'mod': 'mod'}

    while state["token"] in operators:
        name = state["token"]
        fn = operators[name]
        get_token_skip_newline(state)

        if name == '%' and state["tokenType"] == TOKENTYPE['DELIMITER'] and state["token"] != '(':
            # If the expression contains only %, then treat that as /100
            node = OperatorNode('/', 'divide', [node, ConstantNode(100)], False, True)
        else:
            params = [node, parse_unary(state)]
            node = OperatorNode(name, fn, params)

    return node


def parse_unary(state):
    operators = {'-': 'unaryMinus', '+': 'unaryPlus', '~': 'bitNot', 'not': 'not'}

    if state["token"] in operators:
        fn = operators[state["token"]]
        name = state["token"]
        get_token_skip_newline(state)
        params = [parse_unary(state)]
        return OperatorNode(name, fn, params)

    return parse_pow(state)


def parse_pow(state):
    node = parse_left_hand_operators(state)

    if state["token"] == '^' or state["token"] == '.^':
        name = state["token"]
        fn = 'pow' if name == '^' else 'dotPow'
        get_token_skip_newline(state)
        params = [node, parse_unary(state)]  # Go back to unary, we can have '2^-3'
        node = OperatorNode(name, fn, params)

    return node


def parse_left_hand_operators(state):
    node = parse_custom_nodes(state)
    operators = {'!': 'factorial', '\'': 'ctranspose'}

    while state["token"] in operators:
        name = state["token"]
        fn = operators[name]
        get_token(state)
        params = [node]
        node = OperatorNode(name, fn, params)
        node = parse_accessors(state, node)

    return node

def parse_custom_nodes(state):
    params = []

    if state["tokenType"] == TOKENTYPE['SYMBOL'] and state["token"] in state["extraNodes"]:
        CustomNode = state["extraNodes"][state["token"]]
        get_token(state)

        # parse parameters
        if state["token"] == '(':
            params = []
            open_params(state)
            get_token(state)

            if state["token"] != ')':
                params.append(parse_assignment(state))

                # parse a list with parameters
                while state["token"] == ',':
                    get_token(state)
                    params.append(parse_assignment(state))

            if state["token"] != ')':
                raise ValueError('Parenthesis ) expected')

            close_params(state)
            get_token(state)

        # create a new custom node
        return CustomNode(params)

    return parse_symbol(state)


def parse_symbol(state):
    name = state["token"]

    if state["tokenType"] == TOKENTYPE['SYMBOL'] or (state["tokenType"] == TOKENTYPE['DELIMITER'] and state["token"] in NAMED_DELIMITERS):
        get_token(state)

        if name in CONSTANTS:
            node = ConstantNode(CONSTANTS[name])
        elif name in NUMERIC_CONSTANTS:
            node = ConstantNode(numeric(name, 'number'))
        else:
            node = SymbolNode(name)

        # parse function parameters and matrix index
        node = parse_accessors(state, node)
        return node

    return parse_string(state)

def parse_accessors(state, node, types=None):
    while state["token"] in ('(', '[', '.') and (not types or state["token"] in types):
        params = []

        if state["token"] == '(':
            if is_symbol_node(node) or is_accessor_node(node):
                open_params(state)
                get_token(state)

                if state["token"] != ')':
                    params.append(parse_assignment(state))

                    while state["token"] == ',':
                        get_token(state)
                        params.append(parse_assignment(state))

                if state["token"] != ')':
                    raise ValueError('Parenthesis ) expected')

                close_params(state)
                get_token(state)

                node = FunctionNode(node, params)
            else:
                return node
        elif state["token"] == '[':
            open_params(state)
            get_token(state)

            if state["token"] != ']':
                params.append(parse_assignment(state))

                while state["token"] == ',':
                    get_token(state)
                    params.append(parse_assignment(state))

            if state["token"] != ']':
                raise ValueError('Parenthesis ] expected')

            close_params(state)
            get_token(state)

            node = AccessorNode(node, IndexNode(params))
        else:
            get_token(state)

            if state["token"] in NAMED_DELIMITERS or state["tokenType"] == TOKENTYPE['SYMBOL']:
                params.append(ConstantNode(state["token"]))
                get_token(state)

                dot_notation = True
                node = AccessorNode(node, IndexNode(params, dot_notation))

    return node


def parse_string(state):
    if state["token"] in ('"', "'"):
        str_value = parse_string_token(state, state["token"])
        node = ConstantNode(str_value)
        node = parse_accessors(state, node)
        return node

    return parse_matrix(state)

def parse_string_token(state, quote):
    str_value = ''
    while current_character(state) != '' and current_character(state) != quote:
        if current_character(state) == '\\':
            next_character(state)
            char = current_character(state)
            escape_char = ESCAPE_CHARACTERS.get(char)
            if escape_char is not None:
                str_value += escape_char
                state["index"] += 1
            elif char == 'u':
                unicode_char = state["expression"][state["index"] + 1: state["index"] + 5]
                if len(unicode_char) == 4 and all(c in '0123456789abcdefABCDEF' for c in unicode_char):
                    str_value += chr(int(unicode_char, 16))
                    state["index"] += 5
                else:
                    raise ValueError(f'Invalid unicode character \\u{unicode_char}')
            else:
                raise ValueError(f'Bad escape character \\{char}')
        else:
            str_value += current_character(state)
            next_character(state)

    get_token(state)
    if state["token"] != quote:
        raise ValueError(f'End of string {quote} expected')
    get_token(state)

    return str_value


def parse_matrix(state):
    if state["token"] == '[':
        open_params(state)
        get_token(state)

        if state["token"] != ']':
            row = parse_row(state)

            if state["token"] == ';':
                rows = 1
                params = [row]

                while state["token"] == ';':
                    get_token(state)
                    if state["token"] != ']':
                        params.append(parse_row(state))
                        rows += 1

                if state["token"] != ']':
                    raise ValueError('End of matrix ] expected')

                close_params(state)
                get_token(state)

                cols = len(params[0]["items"])
                for r in range(1, rows):
                    if len(params[r]["items"]) != cols:
                        raise ValueError(f'Column dimensions mismatch ({len(params[r]["items"])} !== {cols})')

                array = ArrayNode(params)
            else:
                if state["token"] != ']':
                    raise ValueError('End of matrix ] expected')

                close_params(state)
                get_token(state)
                array = row
        else:
            close_params(state)
            get_token(state)
            array = ArrayNode([])

        return parse_accessors(state, array)

    return parse_object(state)

def parse_row(state):
    params = [parse_assignment(state)]
    length = 1

    while state["token"] == ',':
        get_token(state)
        if state["token"] != ']' and state["token"] != ';':
            params.append(parse_assignment(state))
            length += 1

    return ArrayNode(params)


def parse_object(state):
    if state["token"] == '{':
        open_params(state)
        properties = {}
        while True:
            get_token(state)
            if state["token"] != '}':
                if state["token"] == '"' or state["token"] == "'":
                    key = parse_string_token(state, state["token"])
                elif state["tokenType"] == TOKENTYPE['SYMBOL'] or (state["tokenType"] == TOKENTYPE['DELIMITER'] and state["token"] in NAMED_DELIMITERS):
                    key = state["token"]
                    get_token(state)
                else:
                    raise ValueError('Symbol or string expected as object key')

                if state["token"] != ':':
                    raise ValueError('Colon : expected after object key')
                get_token(state)
                properties[key] = parse_assignment(state)
            if state["token"] != ',':
                break

        if state["token"] != '}':
            raise ValueError('Comma , or bracket } expected after object value')

        close_params(state)
        get_token(state)

        node = ObjectNode(properties)
        node = parse_accessors(state, node)
        return node

    return parse_number(state)


def parse_number(state):
    if state["tokenType"] == TOKENTYPE['NUMBER']:
        number_str = state["token"]
        get_token(state)
        return ConstantNode(numeric(number_str, config.number))

    return parse_parentheses(state)

def parse_parentheses(state):
    node = None

    if state["token"] == '(':
        open_params(state)
        get_token(state)

        node = parse_assignment(state)

        if state["token"] != ')':
            raise ValueError('Parenthesis ) expected')

        close_params(state)
        get_token(state)

        node = ParenthesisNode(node)
        node = parse_accessors(state, node)
        return node

    return parse_end(state)


def parse_end(state):
    if state["token"] == '':
        raise ValueError('Unexpected end of expression')
    else:
        raise ValueError('Value expected')


def col(state):
    return state["index"] - len(state["token"]) + 1


def create_syntax_error(state, message):
    c = col(state)
    error = SyntaxError(message + ' (char ' + str(c) + ')')
    error.char = c

    return error


def create_error(state, message):
    c = col(state)
    error = SyntaxError(message + ' (char ' + str(c) + ')')
    error.char = c

    return error