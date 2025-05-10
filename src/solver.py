import re
from sympy import symbols, Eq, solve, sympify


def solve_equation(equation_str):
    if equation_str == '':
        raise Exception("Empty equation string.")

    # Define the symbol (you can modify to detect more symbols if needed)
    x = symbols('x')

    # Split the equation string at '='
    left_expr, right_expr = equation_str.split('=')

    # Convert strings to sympy expressions
    left = sympify(left_expr)
    right = sympify(right_expr)

    # Create the equation
    equation = Eq(left, right)

    # Solve the equation
    solutions = solve(equation, x)

    return solutions


def process_equation(parsed_equation: list[str]):
    equation_str = ""
    for char in parsed_equation:
        equation_str += char

    # '=' is split when segmenting the image so
    # replacing '--' with '='
    equation_str = equation_str.replace("--", "=")

    # formatting equation for sympy
    # replace all occurrences of xN (x to the power of N) with
    # x^N (using ^ for exponentiation)
    # this will handle cases like x2, x3, x4, etc., and convert them to 
    # x^2, x^3, x^4, etc.
    # if x doesn't have a coefficient, then it adds 1 before *x (x -> 1*x)
    equation_str = re.sub(r'(\d*)x(\d+)', lambda m: (m.group(1)
                          if m.group(1) else '1') + '*x^' + m.group(2),
                          equation_str)

    # # if there's no coefficient and just x (for terms like x or -x),
    # leave it as is ('x' or '-x')
    equation_str = re.sub(r'(?<!\d)x', 'x', equation_str)

    # # handle terms like -8x or 3x correctly, ensuring the sign is maintained
    equation_str = re.sub(r'(?<=\d)x(?!\^)', r'*x', equation_str)

    # remove any characters that shouldn't be there
    equation_str = re.sub(r'[^0-9x+\-\^*/=]', '', equation_str)
    return equation_str


def main():
    equation_str = "-8-x+9--0"
    equation_str = process_equation(equation_str)
    print(equation_str)
    solution = solve_equation(equation_str)
    print(solution)


if __name__ == "__main__":
    main()
