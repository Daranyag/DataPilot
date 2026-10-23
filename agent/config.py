import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

def get_best_models():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing.")
        
    client = genai.Client(api_key=api_key)
    
    flash_lite_model = None
    flash_model = None
    
    # We want to avoid preview models and models that don't support generateContent
    for model in client.models.list():
        if "generateContent" not in model.supported_actions:
            continue
            
        name = model.name
        
        if "preview" in name or "exp" in name or "tts" in name or "audio" in name:
            continue
            
        if "flash-lite" in name:
             if not flash_lite_model or "3." in name:
                 flash_lite_model = name
        elif "flash" in name and "lite" not in name:
             if not flash_model or "3." in name:
                 flash_model = name
                 
    # Default fallbacks if not found
    if not flash_lite_model:
        flash_lite_model = "models/gemini-3.5-flash-lite"
    if not flash_model:
        flash_model = "models/gemini-3.8-flash"
        
    return flash_lite_model, flash_model

def get_model_config():
    try:
        default_model, fallback_model = get_best_models()
    except Exception:
        default_model = "models/gemini-3.5-flash-lite"
        fallback_model = "models/gemini-3.8-flash"
        
    # Read from GEMINI_MODEL env var, use default if not set
    selected_model = os.environ.get("GEMINI_MODEL", default_model)
    return selected_model, fallback_model
