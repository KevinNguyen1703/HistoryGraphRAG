import requests
from langchain_core.language_models.llms import LLM
import json

class CustomLlama(LLM):
    def __init__(self, *args: any, **kwargs: any):
        super().__init__(*args, **kwargs)
    # Implement the required abstract method _llm_type
    @property
    def _llm_type(self):
        return "custom"  # You can return any type string you like

    def _call(self, prompt: str, model: str = "llama3.1:8b", api_url: str = "http://localhost:11434/api/generate",system_prompt=None, history_messages=[], **kwargs) -> str:

        context = []
        if system_prompt:
            context.append(system_prompt)
        context.extend(history_messages)

        payload = {
            "model": model,  # You can specify any model name here (e.g., "mistral", "other_model_name")
            "prompt": prompt,
            "context": context,  # Optional: If you want to add context to the generation
            "stream": False,  # Set to True for streaming if required
        }

        payload.update(kwargs)  # Merge additional parameters
        headers = {"Content-Type": "application/json"}
        response = requests.post(api_url, data=json.dumps(payload), headers=headers)
        response_text = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    response_text += data.get("response", "")
                    if data.get("done", False):
                        break  # Stop when "done": true is encountered
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
        return response_text

