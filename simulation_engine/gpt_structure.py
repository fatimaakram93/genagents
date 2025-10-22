import openai
import time
import base64
import os
import json
import copy # <-- Added import
from typing import List, Sequence, Union, Optional

from .settings import *

openai.api_key = OPENAI_API_KEY


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


# ============================================================================
# ####################### [SECTION 2: SAFE GENERATE] #########################
# ============================================================================

# --- Helper Function for Logging ---
def format_string_for_log(text: str) -> str:
    """Replaces escaped sequences in a string for log readability."""
    if not isinstance(text, str):
        # Attempt to convert non-strings, return original if fails
        try:
            text = str(text)
        except:
            return text
    # Apply replacements for common escapes
    return text.replace('\\n', '\n').replace("\\'", "'").replace('\\"', '"')
# --- End Helper Function ---

def _should_use_responses_api(model: str) -> bool:
  """Return True if this model is served via the Responses API, not Chat Completions."""
  m = (model or "").lower().strip()
  return m in {"o1", "o1-mini", "o1-preview"}


def _uses_completion_token_param(model: str) -> bool:
  """Return True when chat.completions expects max_completion_tokens."""
  m = (model or "").lower().strip()
  return m in {"gpt-5-nano", "gpt-5-mini"}


def _supports_custom_temperature(model: str) -> bool:
  """Return False when the model requires default temperature."""
  m = (model or "").lower().strip()
  return m not in {"gpt-5-nano", "gpt-5-mini"}


def gpt_request(prompt: str,
                model: str = "gpt-4o",
                max_tokens: int = 1500,
                response_format: Optional[dict] = None) -> str:
  """Make a request to OpenAI's GPT model.

  Routes o1* models via the Responses API; others via Chat Completions. gpt-5-mini/nano need completion-token and default temperature handling.
  """
  log_file = os.getenv("OPENAI_DEBUG_LOG", "/Users/fatima.akram/Documents/openai_debug_log.txt")
  # Create log file if it doesn't exist
  if not os.path.exists(log_file):
    open(log_file, 'a').close()

  # Prepare messages for the API call (original, unformatted)
  messages_for_api = [{"role": "user", "content": prompt}]

  # Log the request: pretty-print with json.dumps, then format the resulting string
  with open(log_file, "a") as f:
      f.write("to_server:\n")
      # Get the pretty-printed JSON string first
      json_string = json.dumps(messages_for_api, indent=2, ensure_ascii=False)
      # Now format the entire string for log readability
      formatted_log_string = format_string_for_log(json_string)
      f.write(formatted_log_string + "\n")

  # --- API Call Logic ---
  try:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    if _should_use_responses_api(model):
      # Responses API
      try:
        kwargs = dict(
          model=model,
          input=prompt,
          max_output_tokens=max_tokens,
        )
        if response_format:
          kwargs["text"] = {"format": response_format}
        response = client.responses.create(**kwargs)
      except TypeError:
        # Some client versions don't accept max_output_tokens
        kwargs = {"model": model, "input": prompt}
        if response_format:
          kwargs["text"] = {"format": response_format}
        response = client.responses.create(**kwargs)
      # Prefer convenience property if available
      result = getattr(response, "output_text", None)
      if result is not None and isinstance(result, str) and not result.strip():
        result = None
      if not result:
        # Fallback: attempt to parse from structured output
        try:
          chunks = []
          out = getattr(response, "output", None) or []
          for item in out:
            content_blocks = getattr(item, "content", None) or []
            for block in content_blocks:
              block_text = getattr(block, "text", None)
              if isinstance(block_text, str):
                if block_text.strip():
                  chunks.append(block_text.strip())
                continue
              value = getattr(block_text, "value", None) if block_text is not None else None
              if value:
                chunks.append(str(value).strip())
          result = "\n".join(filter(None, chunks)) if chunks else None
        except Exception:
          result = None
      if not result:
        result = "GENERATION ERROR: empty response"
    else:
      # Chat Completions API
      chat_kwargs = dict(
        model=model,
        messages=messages_for_api, # Send original messages
      )
      if _supports_custom_temperature(model):
        chat_kwargs["temperature"] = 0.7
      if _uses_completion_token_param(model):
        chat_kwargs["max_completion_tokens"] = max_tokens
      else:
        chat_kwargs["max_tokens"] = max_tokens
      if response_format:
        chat_kwargs["response_format"] = response_format
      response = client.chat.completions.create(**chat_kwargs)
      result = response.choices[0].message.content

    # Log the response: format the result string directly
    with open(log_file, "a") as f:
      f.write("from_server:\n")
      f.write(format_string_for_log(result) + "\n-----------\n")
    return result
  except Exception as e:
    error_msg = f"GENERATION ERROR: {str(e)}"
    # Log the error: format the error string directly
    with open(log_file, "a") as f:
      f.write("from_server:\n")
      f.write(format_string_for_log(error_msg) + "\n-----------\n")
    return error_msg


# Note: format_string_for_log defined above gpt_request

def gpt4_vision(messages: List[dict], max_tokens: int = 1500) -> str:
  """Make a request to OpenAI's GPT-4 Vision model."""
  log_file = os.getenv("OPENAI_DEBUG_LOG", "/Users/fatima.akram/Documents/openai_debug_log.txt")
  # Create log file if it doesn't exist
  if not os.path.exists(log_file):
    open(log_file, 'a').close()

  # Log the request: pretty-print with json.dumps, then format the resulting string
  with open(log_file, "a") as f:
      f.write("to_server:\n")
      # Get the pretty-printed JSON string first
      # Use deepcopy to avoid modifying original 'messages' if it contains complex objects
      messages_copy = copy.deepcopy(messages)
      json_string = json.dumps(messages_copy, indent=2, ensure_ascii=False)
      # Now format the entire string for log readability
      formatted_log_string = format_string_for_log(json_string)
      f.write(formatted_log_string + "\n")

  # --- API Call Logic ---
  try:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
      model="gpt-4o",
      messages=messages, # Send original messages
      max_tokens=max_tokens,
      temperature=0.7
    )
    result = response.choices[0].message.content

    # Log the response: format the result string directly
    with open(log_file, "a") as f:
      f.write("from_server:\n")
      f.write(format_string_for_log(result) + "\n-----------\n")
    return result
  except Exception as e:
    error_msg = f"GENERATION ERROR: {str(e)}"
    # Log the error: format the error string directly
    with open(log_file, "a") as f:
      f.write("from_server:\n")
      f.write(format_string_for_log(error_msg) + "\n-----------\n")
    return error_msg


def chat_safe_generate(prompt_input: Union[str, List[str]], 
                       prompt_lib_file: str,
                       gpt_version: str = "gpt-4o", 
                       repeat: int = 1,
                       fail_safe: str = "error", 
                       func_clean_up: callable = None,
                       verbose: bool = False,
                       max_tokens: int = 1500,
                       file_attachment: str = None,
                       file_type: str = None,
                       response_format: Optional[dict] = None) -> tuple:
  """Generate a response using GPT models with error handling & retries."""
  raw_response: Optional[str] = None
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
      raw_response = gpt4_vision(messages, max_tokens)
      response = raw_response

    elif file_type.lower() == 'pdf':
      pdf_text = extract_text_from_pdf_file(file_attachment)
      pdf = f"PDF attachment in text-form:\n{pdf_text}\n\n"
      instruction = generate_prompt(prompt_input, prompt_lib_file)
      prompt = f"{pdf}"
      prompt += f"<End of the PDF attachment>\n=\nTask description:\n{instruction}"
      raw_response = gpt_request(prompt, gpt_version, max_tokens, response_format=response_format)
      response = raw_response

  else:
    prompt = generate_prompt(prompt_input, prompt_lib_file)
    # Treat any response starting with the error prefix as an error and retry
    ERR_PREFIX = "GENERATION ERROR"
    for i in range(repeat):
      raw_response = gpt_request(prompt, model=gpt_version, max_tokens=max_tokens, response_format=response_format)
      response = raw_response
      if not isinstance(response, str) or not response.startswith(ERR_PREFIX):
        break
      time.sleep(2**i)
    else:
      raw_response = response = fail_safe

  if func_clean_up:
    cleaned = func_clean_up(response, prompt=prompt)
    response = cleaned

  if verbose or DEBUG:
    print_run_prompts(prompt_input, prompt, response)

  return response, prompt, prompt_input, fail_safe, raw_response


# ============================================================================
# #################### [SECTION 3: OTHER API FUNCTIONS] ######################
# ============================================================================

def _sanitize_embedding_input(text: str) -> str:
  if not isinstance(text, str) or not text.strip():
    raise ValueError("Input text must be a non-empty string.")
  return text.replace("\n", " ").strip()


def get_text_embeddings(texts: Sequence[str],
                        model: str = "text-embedding-3-small") -> List[List[float]]:
  """Generate embeddings for multiple texts using a single API request."""
  if texts is None:
    raise ValueError("Input texts must not be None.")
  sanitized = [_sanitize_embedding_input(t) for t in texts if isinstance(t, str)]
  if len(sanitized) != len(texts):
    raise ValueError("All inputs must be non-empty strings.")
  if not sanitized:
    return []
  response = openai.embeddings.create(input=sanitized, model=model)
  data = getattr(response, "data", [])
  if len(data) != len(sanitized):
    raise RuntimeError("Embedding response length mismatch.")
  return [item.embedding for item in data]


def get_text_embedding(text: str,
                       model: str = "text-embedding-3-small") -> List[float]:
  """Generate an embedding for the given text using OpenAI's API."""
  embeddings = get_text_embeddings([text], model=model)
  return embeddings[0] if embeddings else []
