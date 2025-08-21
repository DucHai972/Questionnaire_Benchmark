#!/usr/bin/env python3
"""
Robust CSV Parser with Improved Evaluation for Q_Benchmark Results

This script handles malformed CSV files with multi-line content and provides
improved evaluation functions for more accurate assessment of model responses.

Usage:
    python robust_csv_parser_improved.py --file path/to/file.csv
    python robust_csv_parser_improved.py --test-all  # Test all benchmark result files
    python robust_csv_parser_improved.py --re-evaluate  # Re-evaluate with improved functions
"""

import os
import re
import argparse
import glob
from pathlib import Path
from collections import defaultdict

# Import the improved evaluation functions
from improved_evaluation import smart_evaluate


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
        Parse a malformed CSV file and return proper records.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            list: List of dictionaries representing rows
        """
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
                    records.append(record)
            
            print(f"Successfully parsed {len(records)} records from {file_path}")
            return records
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []
    
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
    
    def analyze_file_statistics_improved(self, file_path):
        """
        Analyze file and return statistics using improved evaluation.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            dict: Statistics about the file with improved evaluation
        """
        records = self.parse_file(file_path)
        
        if not records:
            return {
                'total_cases': 0,
                'cases_with_responses': 0,
                'cases_missing_responses': 0,
                'correct_responses_original': 0,
                'correct_responses_improved': 0,
                'incorrect_responses_original': 0,
                'incorrect_responses_improved': 0,
                'accuracy_original': 0.0,
                'accuracy_improved': 0.0,
                'improvement': 0.0
            }
        
        total_cases = len(records)
        cases_with_responses = 0
        correct_responses_original = 0
        correct_responses_improved = 0
        
        for record in records:
            if record['Response'].strip():
                cases_with_responses += 1
                
                # Original evaluation (from CSV)
                if record['Correct'].strip().lower() in ['true', '1', 'yes']:
                    correct_responses_original += 1
                
                # Improved evaluation
                is_correct_improved = smart_evaluate(
                    record['Response'], 
                    record['expected_answer'], 
                    record['task']
                )
                
                if is_correct_improved:
                    correct_responses_improved += 1
        
        accuracy_original = (correct_responses_original / cases_with_responses * 100) if cases_with_responses > 0 else 0.0
        accuracy_improved = (correct_responses_improved / cases_with_responses * 100) if cases_with_responses > 0 else 0.0
        improvement = accuracy_improved - accuracy_original
        
        return {
            'total_cases': total_cases,
            'cases_with_responses': cases_with_responses,
            'cases_missing_responses': total_cases - cases_with_responses,
            'correct_responses_original': correct_responses_original,
            'correct_responses_improved': correct_responses_improved,
            'incorrect_responses_original': cases_with_responses - correct_responses_original,
            'incorrect_responses_improved': cases_with_responses - correct_responses_improved,
            'accuracy_original': accuracy_original,
            'accuracy_improved': accuracy_improved,
            'improvement': improvement
        }


def re_evaluate_all_benchmark_files():
    """Re-evaluate all benchmark result files with improved evaluation."""
    print("Re-evaluating all benchmark result files with improved evaluation...\n")
    
    parser = RobustCSVParserImproved()
    
    # Find all CSV files in benchmark results
    benchmark_dir = "/insight-fast/dnguyen/Q_Benchmark/benchmark_results"
    csv_pattern = os.path.join(benchmark_dir, "**", "*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    print(f"Found {len(csv_files)} CSV files to re-evaluate")
    
    # Group results by model and task
    results_by_model_task = defaultdict(lambda: defaultdict(list))
    
    for csv_file in csv_files:
        rel_path = os.path.relpath(csv_file, benchmark_dir)
        path_parts = rel_path.split(os.sep)
        
        if len(path_parts) >= 3:
            model = path_parts[0]
            dataset = path_parts[1]
            task = path_parts[2]
            filename = path_parts[3] if len(path_parts) > 3 else ""
            
            # Extract task name and format
            if "/" in task:  # Handle nested structure
                task = os.path.basename(csv_file).split('_')[0] + '_' + os.path.basename(csv_file).split('_')[1]
            
            print(f"Re-evaluating: {model}/{dataset}/{filename}")
            
            try:
                stats = parser.analyze_file_statistics_improved(csv_file)
                
                if stats['cases_with_responses'] > 0:
                    results_by_model_task[model][task].append(stats)
                    
                    improvement = stats['improvement']
                    if improvement > 5:  # Show significant improvements
                        print(f"  *** SIGNIFICANT IMPROVEMENT: +{improvement:.1f}% accuracy")
                        print(f"      Original: {stats['accuracy_original']:.1f}% -> Improved: {stats['accuracy_improved']:.1f}%")
                        
            except Exception as e:
                print(f"  Error: {e}")
    
    # Print summary by model and task
    print(f"\n=== RE-EVALUATION SUMMARY ===")
    
    for model in sorted(results_by_model_task.keys()):
        print(f"\nModel: {model}")
        
        model_results = results_by_model_task[model]
        
        for task in sorted(model_results.keys()):
            task_stats = model_results[task]
            
            # Aggregate task statistics
            total_original = sum(s['correct_responses_original'] for s in task_stats)
            total_improved = sum(s['correct_responses_improved'] for s in task_stats)
            total_with_responses = sum(s['cases_with_responses'] for s in task_stats)
            
            if total_with_responses > 0:
                acc_original = total_original / total_with_responses * 100
                acc_improved = total_improved / total_with_responses * 100
                improvement = acc_improved - acc_original
                
                status = "📈 IMPROVED" if improvement > 1 else "➡️  SAME" if abs(improvement) < 1 else "📉 WORSE"
                
                print(f"  {task}: {acc_original:.1f}% -> {acc_improved:.1f}% ({improvement:+.1f}%) {status}")


def main():
    """Main function to run the improved analysis."""
    parser = argparse.ArgumentParser(description='Robust CSV Parser with Improved Evaluation')
    parser.add_argument('--file', type=str, help='Analyze a specific CSV file')
    parser.add_argument('--test-all', action='store_true', help='Test all benchmark result files')
    parser.add_argument('--re-evaluate', action='store_true', help='Re-evaluate all files with improved evaluation')
    
    args = parser.parse_args()
    
    if args.re_evaluate:
        re_evaluate_all_benchmark_files()
    elif args.test_all:
        # Import the original test function
        from robust_csv_parser import test_all_benchmark_files
        test_all_benchmark_files()
    elif args.file:
        csv_parser = RobustCSVParserImproved()
        stats = csv_parser.analyze_file_statistics_improved(args.file)
        
        print(f"File: {args.file}")
        print(f"Total cases: {stats['total_cases']}")
        print(f"Cases with responses: {stats['cases_with_responses']}")
        print(f"Missing responses: {stats['cases_missing_responses']}")
        print(f"\nOriginal Evaluation:")
        print(f"  Correct responses: {stats['correct_responses_original']}")
        print(f"  Accuracy: {stats['accuracy_original']:.1f}%")
        print(f"\nImproved Evaluation:")
        print(f"  Correct responses: {stats['correct_responses_improved']}")
        print(f"  Accuracy: {stats['accuracy_improved']:.1f}%")
        print(f"\nImprovement: {stats['improvement']:+.1f}%")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()