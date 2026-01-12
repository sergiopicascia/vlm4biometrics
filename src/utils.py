import re
from typing import List, Optional


def extract_option_label(
    model_output: str, option_labels: List[str], options: List[str]
) -> Optional[str]:
    """
    Robustly extracts an option label from a given model_output using a
    three-step fallback mechanism.
    """
    model_output_clean = model_output.strip()

    # Define regex patterns to match option labels like (A), A., Option A, etc.
    label_patterns = [
        r"^\s*\b\(?([A-Za-z])\)?\b",  # Matches 'A', '(A)'
        r"^\s*\bOption\s+([A-Za-z])\b",  # Matches 'Option A'
        r"^\s*\b([A-Za-z])\.",  # Matches 'A.'
        r"^\s*\b([A-Za-z]):",  # Matches 'A:'
        r"^\s*\b([A-Za-z])\s+-",  # Matches 'A -'
    ]

    # 1. First, check for a match at the very beginning of the output.
    for pattern in label_patterns:
        match = re.match(pattern, model_output_clean)
        if match:
            label = match.group(1).upper()
            if label in option_labels:
                return label

    # 2. If not found at the beginning, search the entire text for the pattern.
    # We use more general patterns here to find the label anywhere.
    search_patterns = [
        r"\b\(?([A-Za-z])\)?\b",  # Matches 'A', '(A)'
        r"\bOption\s+([A-Za-z])\b",
    ]
    for pattern in search_patterns:
        matches = re.findall(pattern, model_output_clean)
        for label_match in matches:
            label = label_match.upper()
            if label in option_labels:
                return label

    # 3. As a final fallback, check if the option text itself is in the output.
    for idx, option_text in enumerate(options):
        # Check for case-insensitive match
        if option_text.lower() in model_output_clean.lower():
            return option_labels[idx]

    return None
