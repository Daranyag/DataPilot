import ast
from typing import Tuple
from .config import BLOCKED_IMPORTS, BLOCKED_BUILTINS

class SecurityException(Exception):
    pass

def validate_code(code: str) -> Tuple[bool, str]:
    """
    Parses the code into an AST and checks for blocked imports or function calls.
    Returns (True, "") if safe, (False, error_message) if unsafe.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"

    for node in ast.walk(tree):
        # Block dangerous imports (import module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                # check if alias.name or a sub-module starts with blocked
                base_module = alias.name.split('.')[0]
                if base_module in BLOCKED_IMPORTS:
                    return False, f"Security Error: Import of '{alias.name}' is blocked."
        
        # Block dangerous imports (from module import ...)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base_module = node.module.split('.')[0]
                if base_module in BLOCKED_IMPORTS:
                    return False, f"Security Error: Import from '{node.module}' is blocked."

        # Block dangerous built-in functions or property access
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BLOCKED_BUILTINS:
                    return False, f"Security Error: Usage of built-in function '{node.func.id}' is blocked."
        
        # Block access to __builtins__ and getattr
        elif isinstance(node, ast.Name):
            if node.id in {'__builtins__', 'getattr', 'setattr', 'delattr'}:
                return False, f"Security Error: Usage of '{node.id}' is blocked."
        elif isinstance(node, ast.Attribute):
            if node.attr in {'__builtins__', '__class__', '__subclasses__', '__bases__'}:
                return False, f"Security Error: Access to attribute '{node.attr}' is blocked."

    return True, ""
