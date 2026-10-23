import os
import subprocess
import uuid
from typing import Tuple
from .config import SANDBOX_TEMP_DIR, TIMEOUT_SECONDS
from .validator import validate_code

def run_code(code: str, df_path: str) -> Tuple[bool, str]:
    """
    Validates and executes the given Python code in a sandboxed subprocess.
    The script will be provided `df_path` to load the dataset.
    Returns (success: bool, output_or_error: str)
    """
    # 1. Validate the code
    is_safe, validation_msg = validate_code(code)
    if not is_safe:
        return False, validation_msg

    # 2. Prepare the execution script
    # We inject some setup code so the LLM-generated code has access to `df`
    # without having to write the path loading logic every time (though it can).
    script_content = f"""
import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv(r"{df_path}")
except Exception as e:
    print(f"Error loading dataset: {{e}}")

# --- LLM Generated Code ---
{code}
# --------------------------
"""

    # Create a unique temporary file for the script
    script_filename = f"script_{uuid.uuid4().hex}.py"
    script_path = os.path.join(SANDBOX_TEMP_DIR, script_filename)

    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        # 3. Execute in subprocess
        # We run the script in a fresh python process
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=SANDBOX_TEMP_DIR # Restrict working directory
        )

        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr

    except subprocess.TimeoutExpired:
        return False, f"Execution timed out after {TIMEOUT_SECONDS} seconds."
    except Exception as e:
        return False, f"Unexpected execution error: {e}"
    finally:
        # Cleanup script
        if os.path.exists(script_path):
            try:
                os.remove(script_path)
            except:
                pass
