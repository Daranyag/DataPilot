import json

ROUTER_PROMPT = """
You are the Intent Router for DataPilot, an AI Data Science assistant.
Analyze the user's message and extract the core intent and parameters into a valid JSON object.

Allowed intents:
- QUESTION: General questions about the dataset's contents or statistics.
- CHART: Requests to plot or visualize data.
- DATA_CHANGE: Requests to clean, filter, drop columns/rows, scale, or encode data.
- RUN_ALGORITHM: Requests to train or run a single specific machine learning algorithm.
- COMPARE_ALGORITHMS: Requests to train and compare two or more specific algorithms.
- FIT_CHECK: Requests to check if a specific algorithm fits the user's data (e.g., "is Linear Regression good?", "check KNN").
- FIND_BEST: Open-ended requests to automatically find the best algorithm (AutoML).
- UNCLEAR: The request is ambiguous and needs clarification.
- UNSUPPORTED: The request is impossible (e.g., delete files, download internet, unsupported algorithms).

If UNCLEAR, provide a 'message' property with a short clarifying question.
If UNSUPPORTED, provide a 'message' property explaining what is allowed.
For RUN_ALGORITHM, COMPARE_ALGORITHMS, or FIT_CHECK, extract:
- 'algorithms': list of algorithm names mentioned.
- 'target': the target column name, if specified.
- 'features': list of feature columns, if specified.
- 'hyperparameters': dictionary of specified hyperparameters (e.g., {{"n_estimators": 200}}).

Return ONLY the JSON. No markdown ticks, no extra text.

Example JSON output for RUN_ALGORITHM:
{{
  "intent": "RUN_ALGORITHM",
  "algorithms": ["random forest"],
  "target": "salary",
  "features": ["age", "department"],
  "hyperparameters": {{"n_estimators": 200}},
  "message": ""
}}

User Message: {user_message}
"""

def process_intent(data: dict) -> dict:
    from automl.registry import ALGORITHM_ALIASES
    import difflib

    intent = data.get("intent")
    if intent == "FIT_CHECK":
        algorithms = data.get("algorithms", [])
        if not algorithms:
            return {
                "intent": "UNCLEAR",
                "message": "Which algorithm would you like to check?"
            }
        
        target = data.get("target")
        if not target:
            return {
                "intent": "UNCLEAR",
                "message": "I need to know the target column to check if the algorithm fits. What is your target column?"
            }
        
        resolved_algorithms = []
        for algo in algorithms:
            algo_lower = algo.lower().strip()
            if algo_lower in ALGORITHM_ALIASES:
                resolved_algorithms.append(algo_lower)
            else:
                matches = difflib.get_close_matches(algo_lower, list(ALGORITHM_ALIASES.keys()), n=3, cutoff=0.5)
                if matches:
                    return {
                        "intent": "UNSUPPORTED",
                        "message": f"Algorithm '{algo}' is not supported. Did you mean: {', '.join(matches)}?"
                    }
                else:
                    return {
                        "intent": "UNSUPPORTED",
                        "message": f"Algorithm '{algo}' is not supported."
                    }
        data["algorithms"] = resolved_algorithms

    return data

def route_message(user_message: str) -> dict:
    """
    Routes the user message by asking the LLM to classify it and extract parameters.
    Returns a dictionary parsed from the LLM's JSON response.
    """
    from .llm import safe_generate_text
    prompt = ROUTER_PROMPT.format(user_message=user_message)
    
    # We ask for raw JSON to ensure easy parsing
    res = safe_generate_text(prompt, expect_json=True)
    
    if not res.ok:
        return {
            "intent": "UNCLEAR",
            "message": res.user_message
        }
        
    try:
        data = json.loads(res.text)
        # Ensure minimum fields exist
        data.setdefault("intent", "UNCLEAR")
        data.setdefault("message", "Could not parse intent properly.")
        return process_intent(data)
    except json.JSONDecodeError:
        return {
            "intent": "UNCLEAR",
            "message": "I didn't quite understand that. Could you rephrase?",
            "raw_response": res.text
        }
