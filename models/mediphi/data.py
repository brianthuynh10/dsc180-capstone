import ast 
import re 

def extract_response_dict(text: str) -> dict | None:
    """
    Extracts the first Python-style dictionary from a model response string. Very helpful for applying into the LLM_output column of a DataFrame.
    Args:
        text (str): The model response string containing a dictionary.

    Returns:
        dict if successful
        None if extraction or parsing fails
    """
    if not isinstance(text, str):
        return None

    # Match the first {...} block (non-greedy)
    match = re.search(r"\{.*?\}", text, re.DOTALL)

    if not match:
        return None

    dict_str = match.group(0)

    try:
        # Safely parse Python literal
        return ast.literal_eval(dict_str)
    except (SyntaxError, ValueError):
        return None

def parse_report(report: str) -> str:
  """
  We only want the FINDINGS and IMPRESSION parts. Use on bnpp_reports.csv file to clean up the full report.
  """
  match = re.search(r'\bFINDINGS\b.*', report, flags=re.IGNORECASE | re.DOTALL)
  return match.group(0).strip() if match else None