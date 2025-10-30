import os, time, base64, copy, json
import traceback
from typing import List, Union, Optional
import openai
from pathlib import Path
import traceback

from genagents.simulation_engine.settings import *

openai.api_key = OPENAI_API_KEY

# Always write logs to the current working directory.
OPENAI_DEBUG_PATH = Path(os.getenv("OPENAI_DEBUG_LOG", "/Users/fatima.akram/Documents/openai_debug_log.txt")).expanduser()


# ============================================================================
# #######################[SECTION 1: HELPER FUNCTIONS] #######################
# ============================================================================

def print_run_prompts(prompt_input: Union[str, List[str]], 
                      prompt: str, 
                      output: str) -> None:
  print (f"=== START =======================================================")
  print ("~~~ prompt_input    ----------------------------------------------")
  print (prompt_input, "\n")
  print ("~~~ prompt    ----------------------------------------------------")
  print (prompt, "\n")
  print ("~~~ output    ----------------------------------------------------")
  print (output, "\n") 
  print ("=== END ==========================================================")
  print ("\n\n\n")


def generate_prompt(prompt_input: Union[str, List[str]], 
                    prompt_lib_file: str) -> str:
  """Generate a prompt by replacing placeholders in a template file with 
     input."""
  if isinstance(prompt_input, str):
    prompt_input = [prompt_input]
  prompt_input = [str(i) for i in prompt_input]

  with open(prompt_lib_file, "r") as f:
    prompt = f.read()

  for count, input_text in enumerate(prompt_input):
    prompt = prompt.replace(f"!<INPUT {count}>!", input_text)

  if "<commentblockmarker>###</commentblockmarker>" in prompt:
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]

  return prompt.strip()


def _extract_response_text(response) -> str:
  """Best-effort extraction of text content from OpenAI client responses."""
  text = getattr(response, "output_text", None)
  if isinstance(text, str) and text.strip():
    return text.strip()

  output = getattr(response, "output", None)
  if output:
    collected = []
    for item in output:
      content = getattr(item, "content", None)
      if not content and isinstance(item, dict):
        content = item.get("content")
      if not content:
        continue
      for block in content:
        candidate = None
        if hasattr(block, "text"):
          candidate = block.text
        elif hasattr(block, "value"):
          candidate = block.value
        elif isinstance(block, dict):
          candidate = block.get("text") or block.get("value")
        if isinstance(candidate, str):
          collected.append(candidate)
    if collected:
      return "\n".join(collected).strip()

  choices = getattr(response, "choices", None)
  if choices:
    first = choices[0]
    message = getattr(first, "message", None)
    if not message and isinstance(first, dict):
      message = first.get("message")
    if message:
      content = getattr(message, "content", None)
      if not content and isinstance(message, dict):
        content = message.get("content")
      if isinstance(content, str):
        return content.strip()

  return ""


# ============================================================================
# ####################### [SECTION 2: SAFE GENERATE] #########################
# ============================================================================

def gpt_request(prompt: str, 
                model: str = "gpt-4o", 
                max_tokens: Optional[int] = 1500) -> str:
  """Make a request to OpenAI's GPT model."""
  client = openai.OpenAI(api_key=OPENAI_API_KEY)
  request_kwargs = {
    "model": model,
    "input": prompt,
  }
  if model not in {"gpt-5-mini"}:
    request_kwargs["temperature"] = 0.7
  if max_tokens is not None:
    request_kwargs["max_output_tokens"] = max_tokens

  try:
    response = client.responses.create(**request_kwargs)
    text = _extract_response_text(response)
    if text:
      return text
    return "GENERATION ERROR"
  except Exception as e:
    err = f"{type(e).__name__}: {e}"
    try:
      completion_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
      }
      if max_tokens is not None:
        completion_kwargs["max_completion_tokens"] = max_tokens
      if model not in {"gpt-5-mini"}:
        completion_kwargs["temperature"] = 0.7
      response = client.chat.completions.create(**completion_kwargs)
      return response.choices[0].message.content
    except Exception as fallback_exc:
      err = f"{err} | FallbackError {type(fallback_exc).__name__}: {fallback_exc}"
    try:
      with open("openai_debug_log.txt", "a", encoding="utf-8") as _f:
        _f.write(f"[ERROR] model={model}\n{err}\n{traceback.format_exc()}\n\n")
    except Exception:
      pass
    return f"ERROR: {err}"



def gpt4_vision(messages: List[dict], max_tokens: int = 1500) -> str:
  """Make a request to OpenAI's GPT-4 Vision model."""
  try:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    _log_payload("to_server", "gpt-4o", messages)

    formatted_messages = []
    for message in messages:
      role = message.get("role", "user")
      content = message.get("content", "")
      formatted_content = []

      if isinstance(content, str):
        formatted_content.append({"type": "input_text", "text": content})
      elif isinstance(content, list):
        for block in content:
          if isinstance(block, dict):
            block_type = block.get("type")
            if block_type == "text":
              formatted_content.append(
                {"type": "input_text", "text": block.get("text", "")})
            elif block_type in ("image_url", "input_image"):
              image_payload = block.get("image_url") or {}
              if "url" in image_payload:
                formatted_content.append(
                  {"type": "input_image", "image_url": image_payload})
              else:
                base64_data = block.get("image_base64") or image_payload.get("image_base64")
                if base64_data:
                  formatted_content.append(
                    {"type": "input_image", "image_base64": base64_data})
            else:
              formatted_content.append(block)
          else:
            formatted_content.append({"type": "input_text", "text": str(block)})
      else:
        formatted_content.append({"type": "input_text", "text": str(content)})

      formatted_messages.append({"role": role, "content": formatted_content})

    response = client.responses.create(
      model="gpt-4o",
      input=formatted_messages,
      max_output_tokens=max_tokens,
      temperature=0.7
    )
    text = _extract_response_text(response)
    _log_payload("from_server", "gpt-4o", text)
    return text
  except Exception as e:
    try:
      response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_completion_tokens=max_tokens,
        temperature=0.7
      )
      text = response.choices[0].message.content
      _log_payload("from_server", "gpt-4o", text)
      return text
    except Exception as fallback_exc:
      err = f"{type(e).__name__}: {e} | FallbackError {type(fallback_exc).__name__}: {fallback_exc}"
      _log_payload("error", "gpt-4o", err)
      return f"GENERATION ERROR: {err}"


def chat_safe_generate(prompt_input: Union[str, List[str]], 
                       prompt_lib_file: str,
                       gpt_version: str = "gpt-4o", 
                       repeat: int = 1,
                       fail_safe: str = "error", 
                       func_clean_up: callable = None,
                       verbose: bool = False,
                       max_tokens: int = 1500,
                       file_attachment: str = None,
                       file_type: str = None) -> tuple:
  """Generate a response using GPT models with error handling & retries."""
  if file_attachment and file_type:
    prompt = generate_prompt(prompt_input, prompt_lib_file)
    messages = [{"role": "user", "content": prompt}]

    if file_type.lower() == 'image':
      with open(file_attachment, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
      messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "Please refer to the attached image."},
            {"type": "image_url", "image_url": 
              {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
      })
      response = gpt4_vision(messages, max_tokens)

    elif file_type.lower() == 'pdf':
      pdf_text = extract_text_from_pdf_file(file_attachment)
      pdf = f"PDF attachment in text-form:\n{pdf_text}\n\n"
      instruction = generate_prompt(prompt_input, prompt_lib_file)
      prompt = f"{pdf}"
      prompt += f"<End of the PDF attachment>\n=\nTask description:\n{instruction}"
      response = gpt_request(prompt, gpt_version, max_tokens)

  else:
    prompt = generate_prompt(prompt_input, prompt_lib_file)
    for i in range(repeat):
      response = gpt_request(prompt, model=gpt_version)
      if response != "GENERATION ERROR":
        break
      time.sleep(2**i)
    else:
      response = fail_safe

  if func_clean_up:
    response = func_clean_up(response, prompt=prompt)

  if verbose or DEBUG:
    print_run_prompts(prompt_input, prompt, response)

  return response, prompt, prompt_input, fail_safe


# ============================================================================
# #################### [SECTION 3: OTHER API FUNCTIONS] ######################
# ============================================================================

def get_text_embedding(text: str,
                       model: str = "text-embedding-3-small") -> List[float]:
  """Generate an embedding for the given text using OpenAI's API."""
  if not isinstance(text, str) or not text.strip():
    raise ValueError("Input text must be a non-empty string.")

  text = text.replace("\n", " ").strip()
  response = openai.embeddings.create(
    input=[text], model=model).data[0].embedding
  return response
