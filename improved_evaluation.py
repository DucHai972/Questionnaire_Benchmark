#!/usr/bin/env python3
"""
Improved Evaluation Functions for Q-Benchmark

This module provides more sophisticated evaluation functions that handle
various response formats more accurately than simple substring matching.
"""

import re
import ast
import json
from typing import List, Union, Any


def normalize_response_text(text: str) -> str:
    """
    Normalize response text by removing extra whitespace and standardizing format.
    
    Args:
        text (str): Raw response text
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
    
    # Remove extra quotes and whitespace
    text = text.strip().strip('"').strip("'")
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def parse_expected_answer(expected: str) -> Union[List[str], str]:
    """
    Parse expected answer which might be in various formats.

    Args:
        expected (str): Expected answer string

    Returns:
        Union[List[str], str]: Parsed expected answer
    """
    expected = expected.strip()

    # Strip outer quotes first (handles cases like "['1', '2']")
    if expected.startswith('"') and expected.endswith('"'):
        expected = expected[1:-1].strip()
    elif expected.startswith("'") and expected.endswith("'"):
        expected = expected[1:-1].strip()

    # Try to parse as JSON/Python list
    if expected.startswith('[') and expected.endswith(']'):
        try:
            parsed = ast.literal_eval(expected)
            # Convert all elements to strings for consistent comparison
            return [str(item).strip() for item in parsed]
        except:
            try:
                parsed = json.loads(expected)
                return [str(item).strip() for item in parsed]
            except:
                pass
    
    # Check if it's a comma-separated list (without brackets)
    if ',' in expected:
        # Split by comma and clean up each item
        items = [item.strip().strip('"').strip("'") for item in expected.split(',')]
        # Only return as list if we have multiple non-empty items
        if len(items) > 1 and all(item for item in items):
            return items
    
    # If not a list, return as single string
    return expected.strip()


def extract_numbers_and_identifiers(text: str) -> List[str]:
    """
    Extract numbers and identifiers from text.
    
    Args:
        text (str): Input text
        
    Returns:
        List[str]: List of extracted numbers and identifiers
    """
    # Pattern to match numbers, case IDs, and simple identifiers
    patterns = [
        r'\b\d+\b',  # Numbers (standalone)
        r'\bcase_\d+\b',  # Case IDs
        r'\b[A-Za-z]\d+\b',  # Alphanumeric identifiers like A1, B2
        r'\b[A-Za-z]+_\d+\b',  # Underscore identifiers like HOSP_1
        r'\b[A-Za-z]+\d+\b',  # Word+number identifiers like Respondent107
    ]
    
    results = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)

        # For word+number patterns (like Respondent107), only extract the numeric part
        # This prevents "Respondent107" and "107" both being added (causing set mismatch)
        if pattern == r'\b[A-Za-z]+\d+\b':
            for match in matches:
                number_part = re.search(r'\d+', match)
                if number_part:
                    results.append(number_part.group())
        else:
            # For all other patterns, add the full match
            results.extend(matches)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_results = []
    for item in results:
        if item not in seen:
            seen.add(item)
            unique_results.append(item)
    
    return unique_results


def evaluate_rule_based_query(response: str, expected_answer: str) -> bool:
    """
    Improved evaluation for rule-based querying tasks.
    
    This function handles various response formats:
    - Lists of numbers/IDs
    - Comma-separated values
    - Single values
    - Mixed formats
    
    Args:
        response (str): Model response
        expected_answer (str): Expected answer
        
    Returns:
        bool: True if response matches expected answer
    """
    if not response or not response.strip():
        return False
    
    response_normalized = normalize_response_text(response)
    expected_parsed = parse_expected_answer(expected_answer)
    
    # Case 1: Expected answer is a list
    if isinstance(expected_parsed, list):
        # Extract all identifiers from response
        response_items = extract_numbers_and_identifiers(response_normalized)
        
        # Convert expected to strings for comparison
        expected_items = [str(item).strip() for item in expected_parsed]
        
        # Check if sets match (order doesn't matter for rule-based queries)
        response_set = set(response_items)
        expected_set = set(expected_items)
        
        return response_set == expected_set
    
    # Case 2: Expected answer is a single value
    else:
        # Extract identifiers from response
        response_items = extract_numbers_and_identifiers(response_normalized)
        expected_item = str(expected_parsed).strip()
        
        # Check if the expected item is in the response
        return expected_item in response_items


def evaluate_answer_lookup(response: str, expected_answer: str) -> bool:
    """
    Evaluation for answer lookup tasks (more lenient substring matching).
    
    Args:
        response (str): Model response
        expected_answer (str): Expected answer
        
    Returns:
        bool: True if response contains expected answer
    """
    if not response or not response.strip():
        return False
    
    response_normalized = normalize_response_text(response).lower()
    expected_normalized = normalize_response_text(expected_answer).lower()
    
    return expected_normalized in response_normalized


def evaluate_conceptual_aggregation(response: str, expected_answer: str) -> bool:
    """
    Evaluation for conceptual aggregation tasks (exact number matching).
    
    Args:
        response (str): Model response
        expected_answer (str): Expected answer
        
    Returns:
        bool: True if response contains the exact expected number
    """
    if not response or not response.strip():
        return False
    
    # Extract numbers from both response and expected
    response_numbers = re.findall(r'\b\d+\b', response)
    expected_numbers = re.findall(r'\b\d+\b', expected_answer)
    
    if not expected_numbers:
        return False
    
    expected_number = expected_numbers[0]
    return expected_number in response_numbers


def evaluate_respondent_count(response: str, expected_answer: str) -> bool:
    """
    Evaluation for respondent count tasks (exact number matching).
    
    Args:
        response (str): Model response  
        expected_answer (str): Expected answer
        
    Returns:
        bool: True if response contains the exact expected count
    """
    return evaluate_conceptual_aggregation(response, expected_answer)


def evaluate_multi_hop_inference(response: str, expected_answer: str) -> bool:
    """
    Evaluation for multi-hop relational inference tasks.

    Multi-hop inference requires finding relationships and matching all
    related entities exactly (no partial matches).

    Args:
        response (str): Model response
        expected_answer (str): Expected answer

    Returns:
        bool: True if response matches expected answer exactly
    """
    # Use the same logic as answer_reverse_lookup (exact set matching)
    return evaluate_answer_reverse_lookup(response, expected_answer)


def evaluate_answer_reverse_lookup(response: str, expected_answer: str) -> bool:
    """
    Evaluation for answer reverse lookup tasks.

    This function handles various response formats for reverse lookup:
    - "None" responses should match "None" expectations
    - "Respondent107" should match "107"
    - Exact set matching for lists (all items must match)

    Args:
        response (str): Model response
        expected_answer (str): Expected answer

    Returns:
        bool: True if response matches expected answer exactly
    """
    if not response or not response.strip():
        return False
    
    response_normalized = normalize_response_text(response)
    expected_parsed = parse_expected_answer(expected_answer)
    
    # Special case: "None" responses
    if response_normalized.lower() == "none" and str(expected_parsed).lower() == "none":
        return True
    
    # Case 1: Expected answer is a list - check if response matches all expected items exactly
    if isinstance(expected_parsed, list):
        # Extract all identifiers from response
        response_items = extract_numbers_and_identifiers(response_normalized)

        # Convert expected to strings for comparison
        expected_items = [str(item).strip() for item in expected_parsed]

        # For reverse lookup, require exact set match (order doesn't matter)
        response_set = set(response_items)
        expected_set = set(expected_items)

        # Return True if sets match exactly
        return response_set == expected_set
    
    # Case 2: Expected answer is a single value
    else:
        # Extract identifiers from response
        response_items = extract_numbers_and_identifiers(response_normalized)
        expected_item = str(expected_parsed).strip()
        
        # Check if the expected item is in the response
        return expected_item in response_items


def smart_evaluate(response: str, expected_answer: str, task_type: str) -> bool:
    """
    Smart evaluation that chooses the appropriate evaluation function based on task type.
    
    Args:
        response (str): Model response
        expected_answer (str): Expected answer  
        task_type (str): Type of task
        
    Returns:
        bool: True if response is correct according to task-specific evaluation
    """
    if not response or not response.strip():
        return False
    
    task_type = task_type.lower().strip()
    
    evaluation_functions = {
        'rule_based_querying': evaluate_rule_based_query,
        'answer_lookup': evaluate_answer_lookup,
        'answer_reverse_lookup': evaluate_answer_reverse_lookup,
        'conceptual_aggregation': evaluate_conceptual_aggregation,
        'multi_hop_relational_inference': evaluate_multi_hop_inference,
        'respondent_count': evaluate_respondent_count,
    }
    
    # Get the appropriate evaluation function
    eval_func = evaluation_functions.get(task_type, evaluate_answer_lookup)
    
    return eval_func(response, expected_answer)


def test_evaluation_improvements():
    """Test the improved evaluation functions with known cases."""
    
    test_cases = [
        # Rule-based querying examples
        {
            'response': '42, 139, 36',
            'expected': "['42', '139', '36']",
            'task': 'rule_based_querying',
            'should_be_correct': True
        },
        {
            'response': '129',
            'expected': "['129']", 
            'task': 'rule_based_querying',
            'should_be_correct': True
        },
        {
            'response': '88',
            'expected': "['88']",
            'task': 'rule_based_querying', 
            'should_be_correct': True
        },
        {
            'response': '36',
            'expected': "['42', '139', '36']",
            'task': 'rule_based_querying',
            'should_be_correct': False  # Partial match
        },
        # Answer lookup examples
        {
            'response': 'The patient is John Smith',
            'expected': 'John Smith',
            'task': 'answer_lookup',
            'should_be_correct': True
        },
        # Conceptual aggregation examples
        {
            'response': 'There are 25 patients',
            'expected': '25',
            'task': 'conceptual_aggregation',
            'should_be_correct': True
        }
    ]
    
    print("Testing improved evaluation functions...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        result = smart_evaluate(
            test_case['response'], 
            test_case['expected'], 
            test_case['task']
        )
        
        status = "✓ PASS" if result == test_case['should_be_correct'] else "✗ FAIL"
        
        print(f"Test {i}: {status}")
        print(f"  Task: {test_case['task']}")
        print(f"  Response: {test_case['response']}")
        print(f"  Expected: {test_case['expected']}")
        print(f"  Expected result: {test_case['should_be_correct']}")
        print(f"  Actual result: {result}")
        print()


if __name__ == '__main__':
    test_evaluation_improvements()