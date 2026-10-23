import os
import time
import json
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Setup logging
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("DataPilotLLM")
logger.setLevel(logging.ERROR)
if not logger.handlers:
    fh = logging.FileHandler("logs/app.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

def _sanitize_log(msg: str) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if key and len(key) > 4:
        msg = msg.replace(key, f"***{key[-4:]}")
    return msg

@dataclass
class SafeLLMResult:
    ok: bool
    text: str
    error_type: Optional[str] = None
    user_message: Optional[str] = None
    model_used: Optional[str] = None

def get_llm_client() -> Tuple[genai.Client, Optional[str]]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY")
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
        except Exception:
            pass

    if not api_key:
         return None, "GEMINI_API_KEY environment variable is missing. Please set it in .env or your system environment variables."
    try:
        # http_options timeout doesn't exist in the current version of the library shown in docs
        # We'll just rely on the default timeout or pass it to config if available
        client = genai.Client(api_key=api_key)
        return client, None
    except Exception as e:
        logger.error(_sanitize_log(f"Client init error: {str(e)}"))
        return None, "Failed to initialize the AI client. Please check your API key."

def safe_generate_text(prompt: str, system_instruction: str = None, expect_json: bool = False) -> SafeLLMResult:
    """A safe wrapper for all LLM calls that never raises an exception."""
    from .config import get_model_config
    
    client, init_error = get_llm_client()
    if not client:
        return SafeLLMResult(ok=False, text="", error_type="auth", user_message=init_error)
        
    primary_model, fallback_model = get_model_config()
    current_model = primary_model
    
    config = types.GenerateContentConfig(automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    if system_instruction:
        config.system_instruction = system_instruction
        
    if expect_json:
        # We can enforce JSON response mime type if needed, but the prompt says 
        # "retry once with a stricter prompt". We'll handle it via prompt manipulation.
        pass

    max_429_retries = 2
    max_503_retries = 3
    
    retries_429 = 0
    retries_503 = 0
    json_retries = 0
    
    while True:
        try:
            # We don't have a direct timeout arg in generate_content, so we rely on the library's timeout
            # or could use threading/asyncio but we'll stick to synchronous handling as requested.
            response = client.models.generate_content(
                model=current_model,
                contents=prompt,
                config=config
            )
            
            text = response.text
            if not text:
                return SafeLLMResult(ok=False, text="", error_type="empty", user_message="The AI returned an empty response. It might have been blocked by safety filters.", model_used=current_model)
                
            if expect_json:
                # Clean markdown ticks
                cleaned = text.replace("```json", "").replace("```", "").strip()
                try:
                    json.loads(cleaned)
                    return SafeLLMResult(ok=True, text=cleaned, model_used=current_model)
                except json.JSONDecodeError as e:
                    if json_retries < 1:
                        json_retries += 1
                        prompt += "\n\nCRITICAL: You MUST return strictly valid JSON. Do not include any other text."
                        logger.error(f"JSON Decode Error, retrying. Error: {e}, Text: {text}")
                        continue
                    else:
                        logger.error(f"Failed to parse JSON after retry. Text: {text}")
                        return SafeLLMResult(ok=False, text="", error_type="json", user_message="The AI returned invalid data. Please try rephrasing your request.", model_used=current_model)
            
            return SafeLLMResult(ok=True, text=text, model_used=current_model)
            
        except APIError as e:
            err_str = _sanitize_log(str(e))
            code = getattr(e, 'code', None)
            
            if code == 404 or "404" in err_str:
                logger.error(f"Model {current_model} not found (404): {err_str}")
                if current_model == primary_model:
                    current_model = fallback_model
                    continue
                else:
                    return SafeLLMResult(ok=False, text="", error_type="404", user_message="The configured AI models are currently unavailable or retired.", model_used=current_model)
                    
            elif code == 429 or "429" in err_str or "quota" in err_str.lower():
                if retries_429 < max_429_retries:
                    retries_429 += 1
                    sleep_time = 2 ** retries_429
                    logger.error(f"Quota exceeded (429), retrying in {sleep_time}s: {err_str}")
                    time.sleep(sleep_time)
                    continue
                else:
                    logger.error(f"Max retries for 429 reached: {err_str}")
                    return SafeLLMResult(ok=False, text="", error_type="429", user_message="The AI is currently over its usage quota. Please try again in a few minutes.", model_used=current_model)
                    
            elif code in (500, 503) or "503" in err_str or "busy" in err_str.lower():
                if retries_503 < max_503_retries:
                    retries_503 += 1
                    logger.error(f"Service busy ({code}), retrying: {err_str}")
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"Max retries for 503 reached: {err_str}")
                    return SafeLLMResult(ok=False, text="", error_type="503", user_message="The AI service is currently busy. Please try again later.", model_used=current_model)
                    
            elif code == 400 and "instruction is not enabled" in err_str.lower():
                logger.error(f"System instructions not supported by {current_model}. Retrying by prepending to prompt.")
                if config.system_instruction:
                    # Prepend instruction
                    prompt = f"System Instruction:\n{config.system_instruction}\n\nUser:\n{prompt}"
                    config.system_instruction = None
                continue
                
            elif code in (400, 401, 403) or any(x in err_str for x in ["401", "403"]):
                logger.error(f"Auth error ({code}): {err_str}")
                return SafeLLMResult(ok=False, text="", error_type="auth", user_message="Authentication failed. Please check that your GEMINI_API_KEY is valid.", model_used=current_model)
                
            else:
                logger.error(f"Unhandled APIError: {err_str}")
                return SafeLLMResult(ok=False, text="", error_type="api", user_message="An unexpected error occurred while contacting the AI.", model_used=current_model)
                
        except Exception as e:
            err_str = _sanitize_log(str(e))
            logger.error(f"Network or unknown error: {err_str}")
            return SafeLLMResult(ok=False, text="", error_type="unknown", user_message="A network or unexpected error occurred while contacting the AI. Please try again.", model_used=current_model)

def generate_text(prompt: str, system_instruction: str = None) -> str:
    """Legacy wrapper to maintain compatibility while migrating."""
    res = safe_generate_text(prompt, system_instruction)
    if res.ok:
        return res.text
    return res.user_message
