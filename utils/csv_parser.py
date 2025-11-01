#!/usr/bin/env python3
"""
Robust CSV Parser for Q_Benchmark Results

This module handles malformed CSV files with multi-line content.
Provides parsing capabilities for benchmark result CSV files.

Usage:
    from utils.csv_parser import RobustCSVParserImproved
    parser = RobustCSVParserImproved()
    records = parser.parse_file('path/to/file.csv')
"""

import os
import re


class RobustCSVParserImproved:
    """
    Parser that handles malformed CSV files with multi-line content and provides
    improved evaluation capabilities.
    """
    
    def __init__(self):
        self.expected_columns = [
            'case_id', 'task', 'question', 'questionnaire', 
            'expected_answer', 'prompt', 'Response', 'Correct'
        ]
    
    def parse_file(self, file_path):
        """
        Parse a CSV file and return proper records with correct quote unescaping.

        This now uses Python's standard csv module which properly handles CSV escaping.
        Falls back to manual parsing only if standard parsing fails.

        Args:
            file_path (str): Path to the CSV file

        Returns:
            list: List of dictionaries representing rows
        """
        # FIRST: Try standard CSV parser (handles escaping correctly)
        try:
            import csv
            records = []
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Only accept rows with valid case_id
                    if row.get('case_id', '').strip().startswith('case_'):
                        records.append(dict(row))

            if records:
                print(f"Successfully parsed {len(records)} records from {file_path} using standard CSV parser")
                return records
        except Exception as e:
            print(f"Standard CSV parser failed for {file_path}: {e}")
            print(f"Falling back to manual parsing...")

        # FALLBACK: Manual parsing for truly corrupted files
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split by lines and find record boundaries
            lines = content.split('\n')

            # First line should be the header
            if not lines or not lines[0].startswith('case_id'):
                raise ValueError(f"Invalid header in {file_path}")

            header = lines[0].strip()
            expected_header = ','.join(self.expected_columns)
            if header != expected_header:
                print(f"Warning: Header mismatch in {file_path}")
                print(f"Expected: {expected_header}")
                print(f"Found: {header}")

            # Find all case_id lines (record starts)
            record_starts = []
            for i, line in enumerate(lines[1:], 1):  # Skip header
                if line.startswith('case_'):
                    record_starts.append(i)

            if not record_starts:
                print(f"Warning: No case records found in {file_path}")
                return []

            # Parse each record
            records = []
            for i, start_idx in enumerate(record_starts):
                # Determine end of current record
                if i + 1 < len(record_starts):
                    end_idx = record_starts[i + 1]
                else:
                    end_idx = len(lines)

                # Extract record content
                record_lines = lines[start_idx:end_idx]
                record_content = '\n'.join(record_lines).strip()

                # Parse the record
                record = self._parse_record(record_content, file_path, start_idx)
                if record:
                    # IMPORTANT: Unescape CSV quotes
                    for key, value in record.items():
                        if isinstance(value, str):
                            record[key] = self._unescape_csv_quotes(value)
                    records.append(record)

            print(f"Successfully parsed {len(records)} records from {file_path} using manual parser")
            return records

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []

    def _unescape_csv_quotes(self, value):
        """
        Unescape CSV quotes according to RFC 4180.
        In CSV, "" inside a quoted field represents a single ".

        Args:
            value (str): Value potentially with escaped quotes

        Returns:
            str: Value with quotes unescaped
        """
        if not value:
            return value

        # Remove outer quotes if present
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            value = value[1:-1]

        # Replace doubled quotes with single quotes
        value = value.replace('""', '"')

        return value
    
    def _parse_record(self, record_content, file_path, line_num):
        """
        Parse a single record from its raw content.
        
        Args:
            record_content (str): Raw content of the record
            file_path (str): File path for error reporting
            line_num (int): Line number for error reporting
            
        Returns:
            dict or None: Parsed record or None if parsing failed
        """
        try:
            # Look for the response and correct values at the end
            # Pattern: ...content...,Response_value,Correct_value
            
            # Find the last occurrence of patterns that look like responses
            response_pattern = r',([^,]*),\s*(True|False)\s*$'
            match = re.search(response_pattern, record_content)
            
            if match:
                response = match.group(1).strip()
                correct = match.group(2).strip()
                
                # Remove the response and correct parts to get the rest
                content_without_response = record_content[:match.start()]
            else:
                # No response found, might be incomplete
                response = ""
                correct = ""
                content_without_response = record_content
            
            # Now parse the remaining parts
            # The format should be: case_id,task,question,questionnaire,expected_answer,prompt
            
            # Find case_id (first field)
            case_match = re.match(r'^(case_\w+),', content_without_response)
            if not case_match:
                print(f"Warning: Cannot find case_id in record at line {line_num} of {file_path}")
                return None
            
            case_id = case_match.group(1)
            remaining = content_without_response[len(case_id) + 1:]  # +1 for comma
            
            # Parse remaining fields more carefully
            # We know the expected structure, so we can work backwards
            
            # Find task (should be one of the known task types)
            task_types = [
                'answer_lookup', 'answer_reverse_lookup', 'conceptual_aggregation',
                'multi_hop_relational_inference', 'respondent_count', 'rule_based_querying'
            ]
            
            task = None
            for task_type in task_types:
                if remaining.startswith(task_type + ','):
                    task = task_type
                    remaining = remaining[len(task_type) + 1:]
                    break
            
            if not task:
                print(f"Warning: Cannot find valid task in record at line {line_num} of {file_path}")
                return None
            
            # The remaining content is: question,questionnaire,expected_answer,prompt
            # This is tricky because any of these can contain commas and newlines
            
            # Let's try to find the expected_answer by looking for patterns
            # Expected answers are usually simple values, while questionnaires are JSON
            
            # For now, let's use a simpler approach: split and try to reconstruct
            fields = self._extract_remaining_fields(remaining)
            
            if len(fields) >= 4:
                question = fields[0]
                questionnaire = fields[1]
                expected_answer = fields[2]
                prompt = fields[3]
            elif len(fields) == 3:
                # Missing one field, try to infer
                question = fields[0]
                questionnaire = fields[1]
                expected_answer = fields[2]
                prompt = ""
            else:
                print(f"Warning: Cannot parse remaining fields in record at line {line_num} of {file_path}")
                return None
            
            return {
                'case_id': case_id,
                'task': task,
                'question': question,
                'questionnaire': questionnaire,
                'expected_answer': expected_answer,
                'prompt': prompt,
                'Response': response,
                'Correct': correct
            }
            
        except Exception as e:
            print(f"Error parsing record at line {line_num} of {file_path}: {e}")
            return None
    
    def _extract_remaining_fields(self, content):
        """
        Extract the remaining fields: question, questionnaire, expected_answer, prompt
        This is complex due to nested JSON and multi-line content.
        """
        fields = []
        current_field = ""
        in_quotes = False
        brace_depth = 0
        bracket_depth = 0
        
        i = 0
        while i < len(content):
            char = content[i]
            
            if char == '"' and (i == 0 or content[i-1] != '\\'):
                in_quotes = not in_quotes
                current_field += char
            elif not in_quotes:
                if char == '{':
                    brace_depth += 1
                    current_field += char
                elif char == '}':
                    brace_depth -= 1
                    current_field += char
                elif char == '[':
                    bracket_depth += 1
                    current_field += char
                elif char == ']':
                    bracket_depth -= 1
                    current_field += char
                elif char == ',' and brace_depth == 0 and bracket_depth == 0:
                    # This is a field separator
                    fields.append(current_field.strip())
                    current_field = ""
                    i += 1
                    continue
                else:
                    current_field += char
            else:
                current_field += char
            
            i += 1
        
        # Add the last field
        if current_field.strip():
            fields.append(current_field.strip())
        
        return fields
    
