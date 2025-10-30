import json
import re


def extract_first_json_dict(input_str):
  try:
    # Replace curly quotes with standard double quotes
    input_str = (input_str.replace("“", "\"")
                          .replace("”", "\"")
                          .replace("‘", "'")
                          .replace("’", "'"))
    
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
    Robustly extract per-question categorical answers.
    Supports:
      - verbose objects: {"1": {"Response": "C", "Reasoning": "..."}, ...}
      - concise map: {"1":"Yes","2":"No"}
      - regex fallbacks when JSON is slightly malformed.
    """
    try:
        # normalize quotes and try to load JSON if possible
        s = (input_str or "").replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘","'")
        start = s.find("{")
        if start != -1:
            # brace-count to end
            depth, end = 1, start + 1
            while depth > 0 and end < len(s):
                if s[end] == "{": depth += 1
                elif s[end] == "}": depth -= 1
                end += 1
            try:
                obj = json.loads(s[start:end])
            except Exception:
                obj = None
            if isinstance(obj, dict):
                responses, reasonings = [], []
                # numeric keys first
                keys = [k for k in obj.keys() if str(k).isdigit()]
                if keys:
                    for k in sorted(keys, key=lambda x: int(x)):
                        v = obj[k]
                        if isinstance(v, dict):
                            if "Response" in v: responses.append(v["Response"])
                            if "Reasoning" in v: reasonings.append(v["Reasoning"])
                        else:
                            responses.append(str(v))
                    return responses, reasonings
                # flat verbose dict
                for v in obj.values():
                    if isinstance(v, dict) and "Response" in v:
                        responses.append(v["Response"])
                        if "Reasoning" in v: reasonings.append(v["Reasoning"])
                if responses:
                    return responses, reasonings
        # concise numeric map via regex
        num_map = re.findall(r'"\s*(\d+)\s*"\s*:\s*"(Yes|No|[A-Za-z])"', s, flags=re.IGNORECASE)
        if num_map:
            return [v for k, v in sorted(num_map, key=lambda p: int(p[0]))], []
        # legacy verbose regex
        reasonings = re.findall(r'"Reasoning"\s*:\s*"([^"]+)"', s)
        responses  = re.findall(r'"Response"\s*:\s*"([^"]+)"', s)
        return responses, reasonings
    except Exception:
        return [], []


def extract_first_json_dict_numerical(input_str): 
    reasoning_pattern = re.compile(r'"Reasoning":\s*"([^"]+)"')
    response_pattern = re.compile(r'"Response":\s*(\d+\.?\d*)')

    reasonings = reasoning_pattern.findall(input_str)
    responses = response_pattern.findall(input_str)
    return responses, reasonings
