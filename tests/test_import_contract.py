import ast
import importlib
import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

def get_internal_imports():
    root = Path(__file__).resolve().parent.parent
    internal_dirs = ["app", "automl", "agent"]
    imports_to_test = []
    
    for d in internal_dirs:
        dir_path = root / d
        if not dir_path.exists():
            continue
        for p in dir_path.rglob("*.py"):
            with open(p, 'r', encoding='utf-8') as f:
                try:
                    tree = ast.parse(f.read(), str(p))
                except SyntaxError:
                    continue
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module
                    if not module:
                        # Handle relative imports e.g. from .models import ...
                        level = node.level
                        if level > 0:
                            # reconstruct module name based on file path
                            parts = list(p.relative_to(root).parts)[:-1] # strip filename
                            # go up level-1 times
                            for _ in range(level - 1):
                                if parts: parts.pop()
                            module = ".".join(parts)
                    
                    if module and (module.startswith("app") or module.startswith("automl") or module.startswith("agent")):
                        for alias in node.names:
                            # Skip wildcard imports for this strict check
                            if alias.name != '*':
                                imports_to_test.append((module, alias.name, str(p.relative_to(root))))

    return imports_to_test

@pytest.mark.parametrize("module_name, func_name, source_file", get_internal_imports())
def test_import_contract(module_name, func_name, source_file):
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        pytest.fail(f"Could not import module '{module_name}' referenced in '{source_file}'. Error: {e}")
        
    assert hasattr(mod, func_name), f"Module '{module_name}' is missing '{func_name}', which is imported by '{source_file}'."
