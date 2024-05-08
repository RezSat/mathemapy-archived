import re
import math
import cmath
import numpy as np

# Dictionary to store predefined functions
PREDEFINED_FUNCTIONS = {
    'sin': math.sin,
    'cos': math.cos,
    'exp': math.exp,
    'log': math.log,
    # Add more predefined functions as needed
}

# Dictionary to store variable assignments
VARIABLES = {}

def tokenize_expression(expression):
    tokens = []
    current_token = ''
    
    # Define regex patterns for tokens
    equation_pattern = r'='
    semicolon_pattern = r';'
    number_pattern = r'(\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?'
    variable_pattern = r'[a-zA-Z]+'
    operator_pattern = r'[+\-*/^=]'
    function_pattern = r'[a-zA-Z]+\('
    complex_pattern = r'(\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?[+-](\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?[iI]'
    
    # Tokenize the expression
    while expression:
        match = re.match(number_pattern, expression)
        if match:
            tokens.append(('operand', match.group()))
            expression = expression[len(match.group()):]
            continue
        
        match = re.match(variable_pattern, expression)
        if match:
            tokens.append(('operand', match.group()))
            expression = expression[len(match.group()):]
            continue
        
        match = re.match(operator_pattern, expression)
        if match:
            tokens.append(('operator', match.group()))
            expression = expression[len(match.group()):]
            continue
        
        match = re.match(function_pattern, expression)
        if match:
            tokens.append(('function', match.group()[:-1]))
            expression = expression[len(match.group()[:-1]):]
            continue
        
        match = re.match(complex_pattern, expression)
        if match:
            tokens.append(('operand', match.group()))
            expression = expression[len(match.group()):]
            continue

        match = re.match(equation_pattern, expression)
        if match:
            tokens.append(('equality', match.group()))
            expression = expression[len(match.group()):]
            continue

        match = re.match(semicolon_pattern, expression)
        if match:
            tokens.append(('semicolon', match.group()))
            expression = expression[len(match.group()):]
            continue

        if expression[0] == '(' or expression[0] == ')':
            tokens.append(('parenthesis', expression[0]))
            expression = expression[1:]
            continue
        
        if expression[0] == ';':
            tokens.append(('semicolon', expression[0]))
            expression = expression[1:]
            continue
        
        if expression[0].isspace():
            expression = expression[1:]
            continue
        
        raise ValueError(f"Invalid character: {expression[0]}")
    
    return tokens

class ASTNode:
    pass

class OperandNode(ASTNode):
    def __init__(self, value):
        self.value = value

class OperatorNode(ASTNode):
    def __init__(self, operator):
        self.operator = operator
        self.left = None
        self.right = None

class ParenthesisNode(ASTNode):
    def __init__(self, value):
        self.value = value
        self.child = None

class FunctionDefinitionNode(ASTNode):
    def __init__(self, function_name, parameters, expression):
        self.function_name = function_name
        self.parameters = parameters
        self.expression = expression

class ComplexNumber:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag
    
    def __add__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real + other.real, self.imag + other.imag)
        elif isinstance(other, (int, float)):
            return ComplexNumber(self.real + other, self.imag)
        else:
            raise TypeError("Unsupported operand type for +: complex and {}".format(type(other)))
    
    def __sub__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real - other.real, self.imag - other.imag)
        elif isinstance(other, (int, float)):
            return ComplexNumber(self.real - other, self.imag)
        else:
            raise TypeError("Unsupported operand type for -: complex and {}".format(type(other)))
    
    def __mul__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real * other.real - self.imag * other.imag, self.real * other.imag + self.imag * other.real)
        elif isinstance(other, (int, float)):
            return ComplexNumber(self.real * other, self.imag * other)
        else:
            raise TypeError("Unsupported operand type for *: complex and {}".format(type(other)))
    
    def __truediv__(self, other):
        if isinstance(other, ComplexNumber):
            denom = other.real**2 + other.imag**2
            real_part = (self.real * other.real + self.imag * other.imag) / denom
            imag_part = (self.imag * other.real - self.real * other.imag) / denom
            return ComplexNumber(real_part, imag_part)
        elif isinstance(other, (int, float)):
            return ComplexNumber(self.real / other, self.imag / other)
        else:
            raise TypeError("Unsupported operand type for /: complex and {}".format(type(other)))
    
    def __repr__(self):
        return f"({self.real} {'+' if self.imag >= 0 else '-'} {abs(self.imag)}i)"

class EquationNode(ASTNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

class LinearSystemNode(ASTNode):
    def __init__(self, coefficients, constants):
        self.coefficients = coefficients
        self.constants = constants

def parse_equation(tokens):
    lhs_tokens = []
    while tokens[0][1] != '=':
        lhs_tokens.append(tokens.pop(0))
    tokens.pop(0)  # Remove '=' token
    rhs_tokens = tokens[:]
    
    lhs_tree = parse_expression(lhs_tokens)
    rhs_tree = parse_expression(rhs_tokens)
    
    return EquationNode(lhs_tree, rhs_tree)

def parse_expression(tokens):
    # Define operator precedence
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}

    # Helper function to recursively parse the expression
    def parse_expression_rec(tokens, min_precedence=0):
        # Parse the left operand
        left_operand = parse_operand(tokens)

        # Parse subsequent operators and operands
        while tokens and tokens[0][0] == 'operator' and precedence[tokens[0][1]] >= min_precedence:
            operator = tokens.pop(0)[1]

            # Parse the right operand
            right_operand = parse_expression_rec(tokens, min_precedence=precedence[operator] + 1)

            # Create an OperatorNode
            op_node = OperatorNode(operator)
            op_node.left = left_operand
            op_node.right = right_operand

            left_operand = op_node

        return left_operand

    # Call the recursive parsing function
    return parse_expression_rec(tokens)

def parse_statement(tokens):
    if tokens[0][0] == 'function':
        function_name = tokens.pop(0)[1]
        if tokens.pop(0)[1] != '(':
            raise ValueError("Expected '(' after function name")
        parameters = []
        if tokens[0][0] == 'operand':
            parameters.append(tokens.pop(0)[1])
            while tokens[0][1] == ',':
                tokens.pop(0)
                parameters.append(tokens.pop(0)[1])
        if tokens.pop(0)[1] != ')':
            raise ValueError("Expected ')' after function parameters")
        if tokens.pop(0)[1] != '=':
            raise ValueError("Expected '=' after function definition")
        expression = parse_expression(tokens)
        if tokens.pop(0)[1] != ';':
            raise ValueError("Expected ';' after function definition")
        return FunctionDefinitionNode(function_name, parameters, expression)
    raise ValueError("Unknown statement")

def parse_operand(tokens):
    token_type, token_value = tokens.pop(0)
    if token_type == 'operand':
        if token_value in VARIABLES:
            return OperandNode(VARIABLES[token_value])
        return OperandNode(token_value)
    elif token_type == 'parenthesis' and token_value == '(':
        # Handle parentheses
        expression_tree = parse_expression(tokens)
        if tokens.pop(0)[1] != ')':
            raise ValueError("Mismatched parentheses")
        return expression_tree
    elif token_type == 'operator':
        # Handle unary operators
        if token_value == '-':
            return OperatorNode(token_value, right=parse_operand(tokens))
        raise ValueError("Unexpected operator")
    elif token_type == 'function':
        # Handle predefined functions
        function_name = token_value
        if tokens.pop(0)[1] != '(':
            raise ValueError("Expected '(' after function name")
        args_tree = parse_expression(tokens)
        if tokens.pop(0)[1] != ')':
            raise ValueError("Expected ')' after function arguments")
        return FunctionNode(function_name, args_tree)
    else:
        raise ValueError("Unexpected token")

def parse_linear_system(tokens):
    equations = []
    while tokens:
        equation_tokens = []
        while tokens and tokens[0][1] != ';':
            equation_tokens.append(tokens.pop(0))
        tokens.pop(0)  # Remove ';' token
        equation_node = parse_equation(equation_tokens)
        equations.append(equation_node)
    return LinearSystemNode(equations)

def evaluate_statement(statement):
    if isinstance(statement, FunctionDefinitionNode):
        # Add the user-defined function to the dictionary
        VARIABLES[statement.function_name] = statement
    else:
        raise ValueError("Unknown statement")

def evaluate_expression(ast):
    def evaluate_node(node):
        if isinstance(node, OperandNode):
            if isinstance(node.value, str):
                if node.value in VARIABLES:
                    return VARIABLES[node.value]
                else:
                    raise ValueError(f"Undefined variable: {node.value}")
            # Check if the operand is a complex number
            if 'i' in node.value.lower():
                real, imag = map(float, node.value.lower().replace('i', '').split('+'))
                return ComplexNumber(real, imag)
            return float(node.value)
        elif isinstance(node, OperatorNode):
            left_value = evaluate_node(node.left)
            right_value = evaluate_node(node.right)
            if isinstance(left_value, ComplexNumber) or isinstance(right_value, ComplexNumber):
                # If either operand is a complex number, ensure both operands are complex
                if not isinstance(left_value, ComplexNumber):
                    left_value = ComplexNumber(left_value, 0)
                if not isinstance(right_value, ComplexNumber):
                    right_value = ComplexNumber(right_value, 0)
                # Perform the operation for complex numbers
                if node.operator == '+':
                    return left_value + right_value
                elif node.operator == '-':
                    return left_value - right_value
                elif node.operator == '*':
                    return left_value * right_value
                elif node.operator == '/':
                    return left_value / right_value
            else:
                # Perform the operation for real numbers
                if node.operator == '+':
                    return left_value + right_value
                elif node.operator == '-':
                    return left_value - right_value
                elif node.operator == '*':
                    return left_value * right_value
                elif node.operator == '/':
                    if right_value == 0:
                        raise ValueError("Division by zero")
                    return left_value / right_value
        elif isinstance(node, FunctionNode):
            if node.function_name in PREDEFINED_FUNCTIONS:
                args_value = evaluate_node(node.args_tree)
                return PREDEFINED_FUNCTIONS[node.function_name](args_value)
            elif node.function_name in VARIABLES:
                function_def = VARIABLES[node.function_name]
                args_mapping = {param: evaluate_node(arg) for param, arg in zip(function_def.parameters, node.args_tree)}
                return evaluate_expression(substitute_variables(function_def.expression, args_mapping))
            else:
                raise ValueError(f"Undefined function: {node.function_name}")
        else:
            raise ValueError("Invalid AST node type")

    return evaluate_node(ast)

def solve_linear_equation(equation_node):
    lhs = evaluate_expression(equation_node.lhs)
    rhs = evaluate_expression(equation_node.rhs)
    
    if lhs == 0:
        if rhs == 0:
            return "Infinite solutions"  # Equation is of the form 0 = 0
        else:
            return "No solution"  # Equation is of the form 0 = c, where c is nonzero
    else:
        solution = -rhs / lhs
        return solution

def solve_quadratic_equation(equation_node):
    # Extract coefficients a, b, and c from the equation
    a = evaluate_expression(equation_node.lhs.left)
    b = evaluate_expression(equation_node.lhs.right.left)
    c = evaluate_expression(equation_node.rhs)
    
    # Calculate discriminant
    discriminant = b**2 - 4*a*c
    
    # Check discriminant for real roots
    if discriminant > 0:
        root1 = (-b + math.sqrt(discriminant)) / (2*a)
        root2 = (-b - math.sqrt(discriminant)) / (2*a)
        return root1, root2  # Two real roots
    elif discriminant == 0:
        root = -b / (2*a)
        return root, root  # One real root (repeated)
    else:
        # Complex roots
        real_part = -b / (2*a)
        imag_part = math.sqrt(abs(discriminant)) / (2*a)
        root1 = ComplexNumber(real_part, imag_part)
        root2 = ComplexNumber(real_part, -imag_part)
        return root1, root2

def solve_linear_system(system_node):
    equations = system_node.coefficients
    constants = [evaluate_expression(eq.rhs) for eq in equations]
    coefficients = [[evaluate_expression(term) for term in eq.lhs.right.terms] for eq in equations]
    
    # Create the augmented matrix [A | B]
    augmented_matrix = np.column_stack((coefficients, constants))
    
    # Perform Gaussian elimination
    nrows, ncols = augmented_matrix.shape
    for i in range(min(nrows, ncols - 1)):
        pivot_row = i
        while pivot_row < nrows and augmented_matrix[pivot_row, i] == 0:
            pivot_row += 1
        if pivot_row == nrows:
            continue  # Skip if the column has all zeros
        if pivot_row != i:
            augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]  # Swap rows to bring pivot to diagonal
        pivot_val = augmented_matrix[i, i]
        augmented_matrix[i] /= pivot_val  # Make pivot 1
        for j in range(i + 1, nrows):
            multiplier = augmented_matrix[j, i]
            augmented_matrix[j] -= multiplier * augmented_matrix[i]
    
    # Back-substitution
    solution = []
    for i in range(min(nrows, ncols - 1) - 1, -1, -1):
        solution.append(augmented_matrix[i, -1])
        for j in range(i + 1, min(nrows, ncols - 1)):
            solution[-1] -= augmented_matrix[i, j] * solution[-(j - i + 1)]
    
    return solution[::-1]  # Reverse the order of solutions

def differentiate(expression_node, variable):
    if isinstance(expression_node, OperandNode):
        # Base cases
        if expression_node.value == variable:
            return OperandNode('1')  # Derivative of variable with respect to itself is 1
        elif is_constant(expression_node.value):
            return OperandNode('0')  # Derivative of a constant is 0
        elif expression_node.value.startswith(variable + '^'):
            power = expression_node.value.split('^')[1]
            return OperandNode(f"{power}*{variable}^{int(power) - 1}")  # Derivative of x^n is n*x^(n-1)
        else:
            return OperandNode(f"0")  # Treat any other variable as constant
    elif isinstance(expression_node, OperatorNode):
        # Apply differentiation rules based on the operator
        if expression_node.operator == '+':
            left_derivative = differentiate(expression_node.left, variable)
            right_derivative = differentiate(expression_node.right, variable)
            return OperatorNode('+', left_derivative, right_derivative)
        elif expression_node.operator == '-':
            left_derivative = differentiate(expression_node.left, variable)
            right_derivative = differentiate(expression_node.right, variable)
            return OperatorNode('-', left_derivative, right_derivative)
        elif expression_node.operator == '*':
            left = expression_node.left
            right = expression_node.right
            left_derivative = differentiate(left, variable)
            right_derivative = differentiate(right, variable)
            return OperatorNode('+', OperatorNode('*', left_derivative, right), OperatorNode('*', left, right_derivative))
        elif expression_node.operator == '/':
            left = expression_node.left
            right = expression_node.right
            left_derivative = differentiate(left, variable)
            right_derivative = differentiate(right, variable)
            return OperatorNode('/', OperatorNode('-', OperatorNode('*', right, left_derivative), OperatorNode('*', left, right_derivative)), OperatorNode('^', right, OperandNode('2')))
    elif isinstance(expression_node, FunctionNode):
        # Handle differentiation of trigonometric, exponential, and logarithmic functions
        if expression_node.function_name == 'sin':
            arg_derivative = differentiate(expression_node.args_tree, variable)
            return OperatorNode('*', FunctionNode('cos', expression_node.args_tree), arg_derivative)
        elif expression_node.function_name == 'cos':
            arg_derivative = differentiate(expression_node.args_tree, variable)
            return OperatorNode('*', OperatorNode('-', OperandNode('0'), FunctionNode('sin', expression_node.args_tree)), arg_derivative)
        elif expression_node.function_name == 'tan':
            arg_derivative = differentiate(expression_node.args_tree, variable)
            return OperatorNode('*', OperatorNode('^', FunctionNode('sec', expression_node.args_tree), OperandNode('2')), arg_derivative)
        elif expression_node.function_name == 'log':
            arg_derivative = differentiate(expression_node.args_tree, variable)
            return OperatorNode('/', arg_derivative, expression_node.args_tree)
        elif expression_node.function_name == 'exp':
            arg_derivative = differentiate(expression_node.args_tree, variable)
            return OperatorNode('*', FunctionNode('exp', expression_node.args_tree), arg_derivative)
    else:
        raise ValueError("Invalid AST node type")

def integrate(expression_node, variable):
    if isinstance(expression_node, OperandNode):
        # Base cases
        if expression_node.value == variable:
            return OperatorNode('^', OperandNode(variable), OperandNode('2'))  # Integral of x dx is (x^2)/2
        elif is_constant(expression_node.value):
            return OperatorNode('*', OperandNode(expression_node.value), OperandNode(variable))  # Integral of a dx is ax
        elif expression_node.value.startswith(variable + '^'):
            power = expression_node.value.split('^')[1]
            return OperatorNode('/', OperatorNode('^', OperandNode(variable), OperandNode(str(int(power) + 1))), OperandNode(str(int(power) + 1)))  # Integral of x^n dx is x^(n+1)/(n+1)
        else:
            return OperatorNode('*', OperandNode(expression_node.value), OperandNode(variable))  # Treat any other variable as constant
    elif isinstance(expression_node, OperatorNode):
        # Apply integration rules based on the operator
        if expression_node.operator == '+':
            left_integral = integrate(expression_node.left, variable)
            right_integral = integrate(expression_node.right, variable)
            return OperatorNode('+', left_integral, right_integral)
        elif expression_node.operator == '-':
            left_integral = integrate(expression_node.left, variable)
            right_integral = integrate(expression_node.right, variable)
            return OperatorNode('-', left_integral, right_integral)
        elif expression_node.operator == '*':
            left = expression_node.left
            right = expression_node.right
            if is_constant(left.value) or left.value == variable:
                return OperatorNode('*', left, integrate(right, variable))
            elif is_constant(right.value) or right.value == variable:
                return OperatorNode('*', right, integrate(left, variable))
            else:
                raise ValueError("Integration by parts required for product of non-constant functions")
        elif expression_node.operator == '/':
            raise ValueError("Integration of quotient not yet implemented")
    elif isinstance(expression_node, FunctionNode):
        # Handle integration of trigonometric, exponential, and logarithmic functions
        if expression_node.function_name == 'sin':
            return OperatorNode('-', FunctionNode('cos', expression_node.args_tree), FunctionNode('cos', expression_node.args_tree))
        elif expression_node.function_name == 'cos':
            return FunctionNode('sin', expression_node.args_tree)
        elif expression_node.function_name == 'tan':
            return OperatorNode('-', FunctionNode('log', FunctionNode('cos', expression_node.args_tree)), FunctionNode('log', FunctionNode('cos', expression_node.args_tree)))
        elif expression_node.function_name == 'log':
            return OperatorNode('*', OperandNode(variable), FunctionNode('log', expression_node.args_tree))
        elif expression_node.function_name == 'exp':
            return FunctionNode('exp', expression_node.args_tree)
    else:
        raise ValueError("Invalid AST node type")

def integrate_with_limits(expression_node, variable, lower_limit, upper_limit):
    # Integrate the expression with respect to the variable
    integrated_expression = integrate(expression_node, variable)
    
    # Substitute the upper and lower limits into the integrated expression
    integrated_expression_with_limits = substitute_limits(integrated_expression, variable, lower_limit, upper_limit)
    
    # Evaluate the integrated expression with the substituted limits
    result = evaluate_expression(integrated_expression_with_limits)
    
    return result

def substitute_limits(expression_node, variable, lower_limit, upper_limit):
    # Substitute the lower limit into the expression
    expression_with_lower_limit = substitute_variable(expression_node, variable, lower_limit)
    
    # Substitute the upper limit into the expression
    expression_with_limits = substitute_variable(expression_with_lower_limit, variable, upper_limit)
    
    return expression_with_limits

def substitute_variable(expression_node, variable, value):
    if isinstance(expression_node, OperandNode) and expression_node.value == variable:
        # Substitute the value for the variable
        return OperandNode(str(value))
    elif isinstance(expression_node, OperatorNode):
        # Recursively substitute the value in left and right subtrees
        left_substituted = substitute_variable(expression_node.left, variable, value)
        right_substituted = substitute_variable(expression_node.right, variable, value)
        return OperatorNode(expression_node.operator, left_substituted, right_substituted)
    elif isinstance(expression_node, FunctionNode):
        # Recursively substitute the value in function arguments
        args_substituted = substitute_variable(expression_node.args_tree, variable, value)
        return FunctionNode(expression_node.function_name, args_substituted)
    else:
        return expression_node  # Return unchanged if not a variable

def evaluate_limit(expression_node, variable, point):
    # Substitute the point into the expression
    expression_with_point = substitute_variable(expression_node, variable, point)
    
    # Evaluate the expression with the substituted point
    result = evaluate_expression(expression_with_point)
    
    return result

def evaluate_limit_with_lhopital(expression_node, variable, point, step_by_step=False):
    # Substitute the point into the expression
    expression_with_point = substitute_variable(expression_node, variable, point)
    
    # Check if the expression is an indeterminate form
    if is_indeterminate_form(expression_with_point, point):
        # Apply L'Hôpital's rule
        result = apply_lhopitals_rule(expression_node, variable, point, step_by_step)
    else:
        # Evaluate the expression with the substituted point
        result = evaluate_expression(expression_with_point)
    
    return result

def is_indeterminate_form(expression_node, point):
    # Evaluate the expression with the substituted point to check for 0/0 or ∞/∞ forms
    try:
        result = evaluate_expression(expression_node)
        return False  # Expression is not an indeterminate form
    except ZeroDivisionError:
        return True  # Expression is a 0/0 form
    except ValueError:
        return True  # Expression is a ∞/∞ form

def apply_lhopitals_rule(expression_node, variable, point, step_by_step=False):
    # Differentiate the numerator and denominator
    numerator_derivative = differentiate(expression_node.left, variable)
    denominator_derivative = differentiate(expression_node.right, variable)
    
    # Construct the new function with derivatives
    new_function = OperatorNode('/', numerator_derivative, denominator_derivative)
    
    if step_by_step:
        # Provide step-by-step explanation
        print("Applying L'Hôpital's Rule:")
        print("Numerator:", numerator_derivative)
        print("Denominator:", denominator_derivative)
        print("New Function:", new_function)
    
    # Evaluate the new function with the substituted point
    result = evaluate_expression(new_function)
    
    if step_by_step:
        print("Result after L'Hôpital's Rule:", result)
    
    # Check if the result is still an indeterminate form
    if is_indeterminate_form(new_function, point):
        # Repeat L'Hôpital's rule recursively
        result = apply_lhopitals_rule(new_function, variable, point, step_by_step)
    
    return result

def apply_double_angle_identities(expression_node, variable):
    if isinstance(expression_node, FunctionNode):
        if expression_node.function_name == 'sin':
            arg = expression_node.args_tree
            double_arg = OperatorNode('*', OperandNode('2'), arg)
            return OperatorNode('*', OperandNode('2'), FunctionNode('sin', arg), FunctionNode('cos', arg))
        elif expression_node.function_name == 'cos':
            arg = expression_node.args_tree
            return OperatorNode('-', OperatorNode('^', FunctionNode('cos', arg), OperandNode('2')), OperatorNode('^', FunctionNode('sin', arg), OperandNode('2')))
        elif expression_node.function_name == 'tan':
            arg = expression_node.args_tree
            return OperatorNode('/', OperatorNode('*', OperandNode('2'), FunctionNode('tan', arg)), OperatorNode('-', OperandNode('1'), OperatorNode('^', FunctionNode('tan', arg), OperandNode('2'))))
    elif isinstance(expression_node, OperatorNode):
        left = apply_double_angle_identities(expression_node.left, variable)
        right = apply_double_angle_identities(expression_node.right, variable)
        return OperatorNode(expression_node.operator, left, right)
    else:
        return expression_node

# Helper function to check if a value is a constant
def is_constant(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
       
# Test the tokenize_expression function
expression = "3 + 4 * (2 - 1)"
tokens = tokenize_expression(expression)
print(tokens)

# Test the parsing function
parsed_tree = parse_expression(tokens)
print(parsed_tree)

# Test the evaluation function
result = evaluate_expression(parsed_tree)
print("Result:", result)


# Test the enhanced features
expression = "f(x) = x^2 + 1;"
tokens = tokenize_expression(expression)
statement = parse_statement(tokens)
evaluate_statement(statement)

expression = "f(3);"
tokens = tokenize_expression(expression)
parsed_tree = parse_expression(tokens)
result = evaluate_expression(parsed_tree)
print("Result:", result)

# Test complex number operations
a = ComplexNumber(2, 3)
b = ComplexNumber(1, -2)
c = a + b
d = a - b
e = a * b
f = a / b
print("Sum:", c)
print("Difference:", d)
print("Product:", e)
print("Quotient:", f)

# Test tokenization with complex numbers
expression = "3.5+2.7i - 1.2-4i;"
tokens = tokenize_expression(expression)
print(tokens)

# Test evaluation with complex numbers
expression = "3.5+2.7i - 1.2-4i;"
tokens = tokenize_expression(expression)
parsed_tree = parse_expression(tokens)
result = evaluate_expression(parsed_tree)
print("Result:", result)

# Test solving linear equations
expression = "2*x + 5 = 0;"
tokens = tokenize_expression(expression)
equation_node = parse_equation(tokens)
solution = solve_linear_equation(equation_node)
print("Solution:", solution)

# Test solving quadratic equations
expression = "x^2 - 4x + 4 = 0;"
tokens = tokenize_expression(expression)
equation_node = parse_equation(tokens)
solutions = solve_quadratic_equation(equation_node)
print("Solutions:", solutions)

# Test parsing linear systems
expression = "2*x + y = 4; x - 3*y = -6;"
tokens = tokenize_expression(expression)
system_node = parse_linear_system(tokens)
print("System of Equations:", system_node)

# Test solving linear systems
expression = "2*x + y = 4; x - 3*y = -6;"
tokens = tokenize_expression(expression)
system_node = parse_linear_system(tokens)
solution = solve_linear_system(system_node)
print("Solution:", solution)

# Test Example 1
expression = "2*x + y = 4; x - 3*y = -6;"
tokens = tokenize_expression(expression)
system_node = parse_linear_system(tokens)
solution = solve_linear_system(system_node)
print("Solution for Example 1:", solution)

# Test Example 2
expression = "3*x + 2*y - z = 7; 2*x - y + 3*z = -1; x + 3*y + 2*z = 8;"
tokens = tokenize_expression(expression)
system_node = parse_linear_system(tokens)
solution = solve_linear_system(system_node)
print("Solution for Example 2:", solution)


# Test symbolic differentiation
expression = "x^2 + sin(x) + exp(x) - 3/x;"
tokens = tokenize_expression(expression)
parsed_tree = parse_expression(tokens)
derivative_tree = differentiate(parsed_tree, 'x')
print("Original Expression:", expression)
print("Derivative:", derivative_tree)

# Test symbolic integration
expression = "x^2 + sin(x) + exp(x) - 3/x;"
tokens = tokenize_expression(expression)
parsed_tree = parse_expression(tokens)
integral_tree = integrate(parsed_tree, 'x')
print("Original Expression:", expression)
print("Integral:", integral_tree)

# Test symbolic integration with limits
expression = "x^2 + sin(x);"
lower_limit = 0
upper_limit = 1
tokens = tokenize_expression(expression)
parsed_tree = parse_expression(tokens)
result = integrate_with_limits(parsed_tree, 'x', lower_limit, upper_limit)
print("Integral with Limits:", result)

# Test symbolic limits
expression = "(x^2 - 1) / (x - 1);"
point = 1
tokens = tokenize_expression(expression)
parsed_tree = parse_expression(tokens)
result = evaluate_limit(parsed_tree, 'x', point)
print("Limit at Point:", result)


# Test symbolic limits with L'Hôpital's rule
expression = "(sin(x) / x);"
point = 0
tokens = tokenize_expression(expression)
parsed_tree = parse_expression(tokens)
result = evaluate_limit_with_lhopital(parsed_tree, 'x', point, step_by_step=True)
print("Limit with L'Hôpital's Rule:", result)

# Test double-angle identities
expression = "sin(2*x) + cos(2*x);"
tokens = tokenize_expression(expression)
parsed_tree = parse_expression(tokens)
transformed_tree = apply_double_angle_identities(parsed_tree, 'x')
print("Original Expression:", expression)
print("After Applying Double-Angle Identities:", transformed_tree)