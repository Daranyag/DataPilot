import re
import os
from typing import TypedDict, Optional
from .prompts import SYSTEM_PROMPT, RETRY_PROMPT, EXPLAIN_PROMPT
from sandbox.executor import run_code
from sandbox.config import SANDBOX_TEMP_DIR

class AgentState(TypedDict):
    dataset_summary: str
    chat_history: list
    question: str
    dataset_path: str
    generated_code: Optional[str]
    execution_output: Optional[str]
    execution_error: Optional[str]
    retries: int
    max_retries: int
    final_explanation: Optional[str]

def extract_code(text: str) -> str:
    """Extracts python code from markdown block."""
    match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    # Fallback to returning all text if no markdown block found
    return text.strip()

def generate_code_node(state: AgentState) -> AgentState:
    """Generates pandas code to answer the user's question, or handles specific intents."""
    from .llm import safe_generate_text
    from .router import route_message
    
    if state["retries"] == 0:
        # Route message on first try
        intent_data = route_message(state["question"])
        if intent_data.get("intent") == "FIT_CHECK":
            from automl.fit_check import gather_evidence, generate_explanation
            import pandas as pd
            import json
            
            df = pd.read_csv(state["dataset_path"])
            algorithms = intent_data.get("algorithms", [])
            target = intent_data.get("target")
            
            # For simplicity, we just take the first algorithm
            if algorithms and target:
                evidence = gather_evidence(algorithms[0], df, target)
                explanation_data = generate_explanation(evidence)
                
                # Store structured data in the final state for the UI to render
                state["final_explanation"] = explanation_data["text"]
                # We can tuck the evidence into execution_output so the UI can retrieve it
                state["execution_output"] = json.dumps({"fit_check_evidence": evidence})
                state["execution_error"] = None
                state["generated_code"] = "# Fit Check handled by agent"
                return state

    if state["retries"] > 0 and state["execution_error"]:
        prompt = RETRY_PROMPT.format(error=state["execution_error"])
    else:
        history_str = ""
        for msg in state.get("chat_history", [])[-4:]: # include last 4 messages
            if msg["role"] == "user":
                history_str += f"\nUser: {msg['content']}"
            else:
                history_str += f"\nAssistant: {msg.get('content', '')}"
                
        prompt = f"Dataset Summary:\n{state['dataset_summary']}\n\nRecent Conversation History:{history_str}\n\nQuestion: {state['question']}"

    res = safe_generate_text(prompt, system_instruction=SYSTEM_PROMPT)
    if not res.ok:
        # Stop and explain error
        return {**state, "generated_code": None, "final_explanation": res.user_message, "execution_error": "LLM failed"}
        
    import json
    
    # Try to parse the JSON
    text = res.text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
        
    try:
        data = json.loads(text)
    except Exception:
        # Fallback if invalid JSON
        return {**state, "generated_code": None, "execution_error": "Invalid JSON response from LLM.", "retries": state["retries"] + 1}
        
    status = data.get("status", "OK")
    if status != "OK":
        # Handle off-topic, vague, unsafe
        msg = data.get("message", "I cannot answer this question.")
        suggs = data.get("suggestions", [])
        
        explanation = msg
        if suggs:
            explanation += "\n\n**Here are some things you can ask me instead:**\n"
            for s in suggs:
                explanation += f"- {s}\n"
                
        return {**state, "generated_code": None, "final_explanation": explanation, "execution_error": None}
        
    code = data.get("python_code")
    if not code:
        return {**state, "generated_code": None, "execution_error": "No python_code found in JSON.", "retries": state["retries"] + 1}
        
    return {**state, "generated_code": code}

def execute_code_node(state: AgentState) -> AgentState:
    """Executes the generated code in the sandbox."""
    if not state["generated_code"]:
        return {**state, "execution_output": None, "execution_error": "No code generated."}
        
    code = state["generated_code"]
    df_path = state["dataset_path"]
    
    # Ensure chart.png is removed before run if it exists
    chart_path = os.path.join(SANDBOX_TEMP_DIR, "chart.png")
    if os.path.exists(chart_path):
        try:
            os.remove(chart_path)
        except:
            pass

    success, output = run_code(code, df_path)
    
    if success:
        return {**state, "execution_output": output, "execution_error": None}
    else:
        return {**state, "execution_output": None, "execution_error": output, "retries": state["retries"] + 1}

def explain_result_node(state: AgentState) -> AgentState:
    """Explains the execution output in plain English."""
    from .llm import safe_generate_text
    
    if state["final_explanation"]:
         return state # Already explained
         
    if state["retries"] >= state["max_retries"]:
         # Failed after max retries
         error_msg = state["execution_error"] or "Unknown error"
         fallback = f"I tried to write code to answer your question, but it failed. The last error was:\n`{error_msg}`\n\nPlease try rephrasing your question."
         return {**state, "final_explanation": fallback}
         
    prompt = EXPLAIN_PROMPT.format(
        question=state["question"],
        output=state["execution_output"]
    )
    
    res = safe_generate_text(prompt)
    if res.ok:
         explanation = res.text
    else:
         explanation = f"**AI explanation unavailable, showing raw output.**\n\n```text\n{state['execution_output']}\n```"
         
    return {**state, "final_explanation": explanation}
