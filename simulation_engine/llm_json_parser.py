import json
import re
from typing import List


def extract_first_json_dict(input_str):
  try:
    # Replace curly quotes with standard double quotes
    input_str = (input_str.replace("â€", "\"")
                          .replace("â€", "\"")
                          .replace("â€", "\"")
                          .replace("â€", "\""))
    
    # Find the first occurrence of '{' in the input_str
    start_index = input_str.index('{')
    
    # Initialize a count to keep track of open and close braces
    count = 1
    end_index = start_index + 1
    
    # Loop to find the closing '}' for the first JSON dictionary
    while count > 0 and end_index < len(input_str):
        if input_str[end_index] == '{':
            count += 1
        elif input_str[end_index] == '}':
            count -= 1
        end_index += 1
    
    # Extract the JSON substring
    json_str = input_str[start_index:end_index]
    
    # Parse the JSON string into a Python dictionary
    json_dict = json.loads(json_str)
    
    return json_dict
  except ValueError:
    # Handle the case where the JSON parsing fails
    return None


def extract_first_json_dict_categorical(input_str):
    """
    Extract responses and reasonings from categorical LLM responses.
    Supports:
      (A) top-level arrays: {"responses":[...], "reasonings":[...]}
      (B) concise numeric map: {"1":"Yes","2":"No",...}
      (C) verbose per-item dicts: {"1":{"Response":"Yes","Reasoning":"..."} , ...}
    """
    try:
        # First, try to parse the entire JSON structure
        json_match = re.search(r'\{.*\}', input_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                # Parse the full JSON
                full_json = json.loads(json_str)

                responses: List[str] = []
                reasonings: List[str] = []

                # (A) top-level arrays
                if isinstance(full_json, dict) and "responses" in full_json and isinstance(full_json["responses"], list):
                    responses = [str(x) for x in full_json["responses"]]
                    if isinstance(full_json.get("reasonings"), list):
                        reasonings = [str(x) for x in full_json.get("reasonings", [])]
                    return responses, reasonings

                # (B) concise numeric map: keys are "1","2",...
                if isinstance(full_json, dict):
                    numeric_keys = [k for k in full_json.keys() if str(k).isdigit()]
                    if numeric_keys:
                        for k in sorted(numeric_keys, key=lambda x: int(x)):
                            v = full_json[k]
                            if isinstance(v, dict):
                                # (C) verbose per-item dicts
                                if "Response" in v:
                                    responses.append(v["Response"])
                                if "Reasoning" in v:
                                    reasonings.append(v["Reasoning"])
                            else:
                                # concise string like "Yes"/"No"
                                responses.append(str(v))
                        return responses, reasonings

                # (C) purely verbose object without numeric keys – scan values
                if isinstance(full_json, dict):
                    for v in full_json.values():
                        if isinstance(v, dict):
                            if "Response" in v:
                                responses.append(v["Response"])
                            if "Reasoning" in v:
                                reasonings.append(v["Reasoning"])
                    if responses:
                        return responses, reasonings

            except json.JSONDecodeError:
                # Fall back to regex if JSON parsing fails
                pass

        # Fallbacks:
        # (B) concise numeric map via regex
        num_map_pat = re.compile(r'"\s*(\d+)\s*"\s*:\s*"(Yes|No)"', re.IGNORECASE)
        num_pairs = num_map_pat.findall(input_str)
        if num_pairs:
            responses = [v for k, v in sorted(num_pairs, key=lambda p: int(p[0]))]
            return responses, []

        # (C) legacy verbose via regex
        reasoning_pattern = r'"Reasoning"\s*:\s*"([^"]+)"'
        response_pattern = r'"Response"\s*:\s*"([^"]+)"'

        reasonings = re.findall(reasoning_pattern, input_str)
        responses = re.findall(response_pattern, input_str)

        return responses, reasonings

    except Exception:
        return [], []


def extract_first_json_dict_numerical(input_str): 
  reasoning_pattern = re.compile(r'"Reasoning":\s*"([^"]+)"')
  response_pattern = re.compile(r'"Response":\s*(\d+\.?\d*)')

  reasonings = reasoning_pattern.findall(input_str)
  responses = response_pattern.findall(input_str)
  return responses, reasonings
