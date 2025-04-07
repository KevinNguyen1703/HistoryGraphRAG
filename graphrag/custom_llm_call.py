from openai import AsyncOpenAI
from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage
import requests
import json
from functools import wraps
import asyncio
import os
from openai import OpenAI


def limit_async_func_call(max_size: int, waitting_time: float = 0.001):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro
def ollama_complete(
    prompt, system_prompt=None, history_messages=[], model="mistral", **kwargs
) -> str:
    """
    Generates a response using the Ollama model, similar to `gpt_4o_mini_complete`.

    :param prompt: The user input prompt.
    :param system_prompt: An optional system message to guide the response.
    :param history_messages: A list of past messages for context.
    :param model: The name of the Ollama model (default: "mistral").
    :param kwargs: Additional parameters.
    :return: The generated response as a string.
    """
    url = "http://localhost:11434/api/generate"

    # Constructing the complete context with system prompt and history
    context = []
    if system_prompt:
        context.append(system_prompt)
    context.extend(history_messages)

    payload = {
        "model": model,  # Default is "mistral"
        "prompt": prompt,
        "context": context,
        "stream": False,  # Set to True if you want a streaming response
    }
    payload.update(kwargs)  # Merge additional parameters

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    # Making the request
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


async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
    response_text = response.choices[0].message.content
    print("[+]      GPT prompt: "+ prompt)
    print("[+]      GPT content: "+ str(system_prompt))
    print("[+]      GPT response: "+ response_text)
    return response_text


async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.getenv("OPENAI_API_KEY"),
)
def call_openai(content):
    """Gửi yêu cầu đến OpenAI API."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}]
    )
    return response.choices[0].message.content.strip()