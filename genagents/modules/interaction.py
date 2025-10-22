import math
import sys
import datetime
import random
import string
import re

from numpy import dot
from numpy.linalg import norm

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from simulation_engine.settings import * 
from simulation_engine.global_methods import *
from simulation_engine.gpt_structure import *
from simulation_engine.llm_json_parser import *


def _main_agent_desc(agent, anchor): 
  agent_desc = ""
  agent_desc += f"Self description: {agent.get_self_description()}\n==\n"
  agent_desc += f"Other observations about the subject:\n\n"

  retrieved = agent.memory_stream.retrieve([anchor], 0, n_count=120)
  if len(retrieved) == 0:
    return agent_desc
  nodes = list(retrieved.values())[0]
  for node in nodes:
    agent_desc += f"{node.content}\n"
  return agent_desc


def _utterance_agent_desc(agent, anchor): 
  agent_desc = ""
  agent_desc += f"Self description: {agent.get_self_description()}\n==\n"
  agent_desc += f"Other observations about the subject:\n\n"

  retrieved = agent.memory_stream.retrieve([anchor], 0, n_count=120)
  if len(retrieved) == 0:
    return agent_desc
  
  nodes = list(retrieved.values())[0]
  for node in nodes:
    agent_desc += f"{node.content}\n"
  return agent_desc


def run_gpt_generate_categorical_resp(
  agent_desc, 
  questions,
  prompt_version="1",
  gpt_version="GPT4o",  
  verbose=False):

  def create_prompt_input(agent_desc, questions):
    str_questions = ""
    for key, val in questions.items(): 
      str_questions += f"Q: {key}\n"
      str_questions += f"Option: {val}\n\n"
    str_questions = str_questions.strip()
    return [agent_desc, str_questions]

  def _func_clean_up(gpt_response, prompt=""): 
    responses, reasonings = extract_first_json_dict_categorical(gpt_response)
    ret = {"responses": responses, "reasonings": reasonings}
    return ret

  def _get_fail_safe():
    return None

  if len(questions) > 1: 
    # Use the concise prompt for batch requests to avoid token limits
    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/categorical_resp/batch_v1_concise.txt" 
  else: 
    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/categorical_resp/singular_v1.txt" 

  prompt_input = create_prompt_input(agent_desc, questions) 
  fail_safe = _get_fail_safe() 
  model_name = (gpt_version or "").lower()
  # gpt-5-nano behaves better with free-form text responses; skip JSON mode there.
  response_format_arg = None if "gpt-5-nano" in model_name else {"type": "json_object"}

  output, prompt, prompt_input, fail_safe, raw_response = chat_safe_generate(
    prompt_input, prompt_lib_file, gpt_version, 1, fail_safe, 
    _func_clean_up, verbose, response_format=response_format_arg)

  def _options_for_question() -> list[str]:
    if not questions:
      return []
    # Deterministic iteration: use first key sorted alphabetically to keep behaviour stable
    key = sorted(questions.keys())[0]
    vals = questions.get(key, [])
    return [str(v) for v in vals] if isinstance(vals, (list, tuple)) else [str(vals)]

  def _guess_response(raw_text: str) -> str | None:
    if not raw_text:
      return None
    candidates = {"c", "d"}
    options = _options_for_question()

    def from_index(idx: int) -> str | None:
      if 0 <= idx < len(options):
        token = options[idx].strip()
        if token:
          head = token[0].upper()
          if head in {"C", "D"}:
            return head
          if token.upper() in {"C", "D"}:
            return token.upper()
      return None

    text = raw_text.strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Direct JSON-like pattern: "Response": "C"
    resp_match = re.search(r'"?response"?\s*[:\-]?\s*"?([cd])"?', raw_text, re.IGNORECASE)
    if resp_match:
      return resp_match.group(1).upper()

    # Scan lines for explicit answer markers (bottom-up to favour concluding statement)
    for idx in range(len(lines) - 1, -1, -1):
      line = lines[idx]
      low = line.lower()
      if any(token in low for token in ("option interpretation", "option choice", "option:")):
        continue
      if low.startswith("reasoning"):
        continue
      stripped = line.strip().lower()
      key_match = re.match(r'"([cd])"\s*:', stripped)
      if key_match:
        return key_match.group(1).upper()
      if low.startswith("output") and idx + 1 < len(lines):
        next_line = lines[idx + 1]
        guess = _guess_response(next_line)
        if guess:
          return guess
      cleaned_letters = re.findall(r'\b([cd])\b', low)
      for letter in cleaned_letters:
        return letter.upper()
      # Check for Option references like "Option 1"
      opt_match = re.search(r'option\s*([12])', low)
      if opt_match:
        mapped = from_index(int(opt_match.group(1)) - 1)
        if mapped:
          return mapped
      # Check for digit-only line implying choice index
      digit_match = re.match(r'^[12](?:[^0-9]|$)', line)
      if digit_match:
        mapped = from_index(int(line[0]) - 1)
        if mapped:
          return mapped
      # Single-character line like "C" or "D"
      if len(line) == 1 and line.upper() in {"C", "D"}:
        return line.upper()

    # Fallback: examine leading character of entire response
    if text:
      # Try inline references such as '"c"' or '"d"' scattered in text
      inline_key = re.search(r'"([cd])"\s*:', text.lower())
      if inline_key:
        return inline_key.group(1).upper()
      inline_digit = re.search(r'\b([12])\b', text)
      if inline_digit:
        mapped = from_index(int(inline_digit.group(1)) - 1)
        if mapped:
          return mapped
      first = text[0]
      if first.upper() in {"C", "D"}:
        return first.upper()
      if first in "12":
        mapped = from_index(int(first) - 1)
        if mapped:
          return mapped

    return None

  normalized = {"responses": [], "reasonings": []}
  if isinstance(output, dict):
    normalized["responses"] = [str(x) for x in output.get("responses", []) if str(x).strip()]
    normalized["reasonings"] = [str(x) for x in output.get("reasonings", []) if str(x).strip()]

  if not normalized["responses"]:
    guess = _guess_response(raw_response or "")
    if guess:
      normalized["responses"] = [guess]

  return normalized, [normalized, prompt, prompt_input, fail_safe, raw_response]


def categorical_resp(agent, questions): 
  anchor = " ".join(list(questions.keys()))
  agent_desc = _main_agent_desc(agent, anchor)
  return run_gpt_generate_categorical_resp(
           agent_desc, questions, "1", LLM_VERS)[0]


def run_gpt_generate_numerical_resp(
  agent_desc, 
  questions, 
  float_resp,
  prompt_version="1",
  gpt_version="GPT4o",  
  verbose=False):

  def create_prompt_input(agent_desc, questions, float_resp):
    str_questions = ""
    for key, val in questions.items(): 
      str_questions += f"Q: {key}\n"
      str_questions += f"Range: {str(val)}\n\n"
    str_questions = str_questions.strip()

    if float_resp: 
      resp_type = "float"
    else: 
      resp_type = "integer"
    return [agent_desc, str_questions, resp_type]

  def _func_clean_up(gpt_response, prompt=""): 
    responses, reasonings = extract_first_json_dict_numerical(gpt_response)
    ret = {"responses": responses, "reasonings": reasonings}
    return ret

  def _get_fail_safe():
    return None

  if len(questions) > 1: 
    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/numerical_resp/batch_v1.txt" 
  else: 
    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/numerical_resp/singular_v1.txt" 

  prompt_input = create_prompt_input(agent_desc, questions, float_resp) 
  fail_safe = _get_fail_safe() 

  output, prompt, prompt_input, fail_safe, _ = chat_safe_generate(
    prompt_input, prompt_lib_file, gpt_version, 1, fail_safe, 
    _func_clean_up, verbose)

  if float_resp: 
    output["responses"] = [float(i) for i in output["responses"]]
  else: 
    output["responses"] = [int(i) for i in output["responses"]]

  return output, [output, prompt, prompt_input, fail_safe]


def numerical_resp(agent, questions, float_resp): 
  anchor = " ".join(list(questions.keys()))
  agent_desc = _main_agent_desc(agent, anchor)
  return run_gpt_generate_numerical_resp(
           agent_desc, questions, float_resp, "1", LLM_VERS)[0]


def run_gpt_generate_utterance(
  agent_desc, 
  str_dialogue,
  context,
  prompt_version="1",
  gpt_version="GPT4o",  
  verbose=False):

  def create_prompt_input(agent_desc, str_dialogue, context):
    return [agent_desc, context, str_dialogue]

  def _func_clean_up(gpt_response, prompt=""): 
    utterance = extract_first_json_dict(gpt_response)["utterance"]
    return utterance

  def _get_fail_safe():
    return None

  prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/utternace/utterance_v1.txt" 

  prompt_input = create_prompt_input(agent_desc, str_dialogue, context) 
  fail_safe = _get_fail_safe() 

  output, prompt, prompt_input, fail_safe, _ = chat_safe_generate(
    prompt_input, prompt_lib_file, gpt_version, 1, fail_safe, 
    _func_clean_up, verbose)

  return output, [output, prompt, prompt_input, fail_safe]


def utterance(agent, curr_dialogue, context): 
  str_dialogue = ""
  for row in curr_dialogue:
    str_dialogue += f"[{row[0]}]: {row[1]}\n"
  str_dialogue += f"[{agent.get_fullname()}]: [Fill in]\n"

  anchor = str_dialogue
  agent_desc = _utterance_agent_desc(agent, anchor)
  return run_gpt_generate_utterance(
           agent_desc, str_dialogue, context, "1", LLM_VERS)[0]

##  Ask function.
def run_gpt_generate_ask(
    agent_desc,
    questions,
    prompt_version="1",
    gpt_version="GPT4o",
    verbose=False):

    def create_prompt_input(agent_desc, questions):
        str_questions = ""
        i = 1
        for q in questions:
            str_questions += f"Q{i}: {q['question']}\n"
            str_questions += f"Type: {q['response-type']}\n"
            if q['response-type'] == 'categorical':
                str_questions += f"Options: {', '.join(q['response-options'])}\n"
            elif q['response-type'] in ['int', 'float']:
                str_questions += f"Range: {q['response-scale']}\n"
            elif q['response-type'] == 'open':
                char_limit = q.get('response-char-limit', 200)
                str_questions += f"Character Limit: {char_limit}\n"
            str_questions += "\n"
            i += 1
        return [agent_desc, str_questions.strip()]

    def _func_clean_up(gpt_response, prompt=""):
        responses = extract_first_json_dict(gpt_response)
        return responses

    def _get_fail_safe():
        return None

    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/ask/batch_v1.txt"

    prompt_input = create_prompt_input(agent_desc, questions)
    fail_safe = _get_fail_safe()

    output, prompt, prompt_input, fail_safe, _ = chat_safe_generate(
        prompt_input, prompt_lib_file, gpt_version, 1, fail_safe,
        _func_clean_up, verbose)

    return output, [output, prompt, prompt_input, fail_safe]
