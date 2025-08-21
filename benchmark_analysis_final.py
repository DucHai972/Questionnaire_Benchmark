#!/usr/bin/env python3
"""
Final Benchmark Results Analysis Script with Improved Evaluation and CSV Updates

This script analyzes the benchmark results from Q_Benchmark/benchmark_results/
using both the robust CSV parser for proper file handling AND improved evaluation
functions for more accurate assessment of model responses.

NEW: This script now WRITES the improved evaluation results back to the CSV files,
updating the "Correct" column with more accurate task-specific evaluations.

This provides the most accurate and reliable analysis available while also
updating the source CSV files with improved evaluation results.

Usage:
  python benchmark_analysis_final.py --model gemini-2.5-flash  # Analysis + CSV updates
  python benchmark_analysis_final.py --model gpt-5-mini       # Different model  
  python benchmark_analysis_final.py --list                   # List available models
"""

import os
import csv
import glob
import argparse
from pathlib import Path
from collections import defaultdict

# Import the robust parser with improved evaluation
from robust_csv_parser_improved import RobustCSVParserImproved
from improved_evaluation import smart_evaluate
import csv


def _write_records_to_csv(csv_file, records):
    """
    Write records back to a CSV file with proper formatting.
    
    Args:
        csv_file (str): Path to the CSV file
        records (list): List of record dictionaries to write
    """
    if not records:
        return
    
    fieldnames = ["case_id", "task", "question", "questionnaire", 
                 "expected_answer", "prompt", "Response", "Correct"]
    
    try:
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                # Ensure all required fields are present
                row = {}
                for field in fieldnames:
                    row[field] = record.get(field, '')
                writer.writerow(row)
    except Exception as e:
        print(f"Error writing to {csv_file}: {e}")


def analyze_benchmark_results_final(base_path):
    """
    Analyze benchmark results using robust parsing and improved evaluation.
    
    Args:
        base_path (str): Path to the benchmark results directory
    
    Returns:
        tuple: (datasets, tasks, data_formats, results) - Analysis results
    """
    
    # Find all CSV files in the benchmark results
    csv_pattern = os.path.join(base_path, "**", "*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    print(f"Found {len(csv_files)} CSV files to analyze")
    
    # Initialize the robust parser with improved evaluation
    parser = RobustCSVParserImproved()
    
    # Initialize dictionaries to store results
    results = {}
    
    # Data formats and tasks we expect to find
    data_formats = ['html', 'json', 'md', 'ttl', 'txt', 'xml']
    tasks = set()
    datasets = set()
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            # Extract information from file path
            rel_path = os.path.relpath(csv_file, base_path)
            path_parts = rel_path.split(os.sep)
            
            if len(path_parts) >= 3:
                dataset = path_parts[0]
                task = path_parts[1]
                filename = path_parts[2]
                
                # Extract data format from filename
                data_format = None
                for fmt in data_formats:
                    if f"_{fmt}_converted_prompts.csv" in filename:
                        data_format = fmt
                        break
                
                if data_format:
                    datasets.add(dataset)
                    tasks.add(task)
                    
                    # Use the robust parser with improved evaluation
                    print(f"Processing: {dataset}/{task}/{data_format}")
                    
                    # Parse the file using the robust parser
                    records = parser.parse_file(csv_file)
                    
                    if records:
                        total_count = len(records)
                        correct_count = 0
                        records_updated = False
                        
                        for record in records:
                            # Only evaluate records that have responses
                            if record['Response'].strip():
                                # Use improved evaluation instead of CSV "Correct" column
                                is_correct = smart_evaluate(
                                    record['Response'], 
                                    record['expected_answer'], 
                                    record['task']
                                )
                                
                                # Update the Correct column with improved evaluation
                                new_correct_value = "True" if is_correct else "False"
                                if record.get('Correct') != new_correct_value:
                                    record['Correct'] = new_correct_value
                                    records_updated = True
                                
                                if is_correct:
                                    correct_count += 1
                            else:
                                # No response, mark as incorrect
                                if record.get('Correct') != "False":
                                    record['Correct'] = "False"
                                    records_updated = True
                        
                        # Write back to CSV file if any records were updated
                        if records_updated:
                            _write_records_to_csv(csv_file, records)
                            print(f"  -> Updated Correct column in {csv_file}")
                        
                        # Store results
                        key = (dataset, task, data_format)
                        results[key] = {
                            'correct': correct_count,
                            'total': total_count,
                            'percentage': (correct_count / total_count * 100) if total_count > 0 else 0
                        }
                        
                        print(f"  -> {correct_count}/{total_count} correct ({results[key]['percentage']:.1f}%)")
                    else:
                        print(f"  -> No records found")
                        
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            continue
    
    # Convert to sorted lists for consistent ordering
    datasets = sorted(list(datasets))
    tasks = sorted(list(tasks))
    data_formats = sorted(data_formats)
    
    print(f"\nFound datasets: {datasets}")
    print(f"Found tasks: {tasks}")
    print(f"Found data formats: {data_formats}")
    
    return datasets, tasks, data_formats, results


def create_tables(datasets, tasks, data_formats, results):
    """
    Create count and percentage tables from the results.
    
    Args:
        datasets, tasks, data_formats: Lists of dataset, task, and format names
        results: Dictionary with analysis results
        
    Returns:
        tuple: (count_table, percentage_table) as lists of lists
    """
    
    # Initialize result tables with tasks as columns (aggregated across all datasets)
    count_table = []
    percentage_table = []
    
    # Build tables with data formats as rows and tasks as columns
    for data_format in data_formats:
        count_row = []
        percentage_row = []
        
        for task in tasks:
            # Aggregate results across all datasets for this task and format
            total_correct = 0
            total_questions = 0
            
            for dataset in datasets:
                key = (dataset, task, data_format)
                if key in results:
                    total_correct += results[key]['correct']
                    total_questions += results[key]['total']
            
            if total_questions > 0:
                count_row.append(f"{total_correct}/{total_questions}")
                percentage = (total_correct / total_questions * 100)
                percentage_row.append(f"{percentage:.1f}%")
            else:
                count_row.append("N/A")
                percentage_row.append("N/A")
        
        count_table.append(count_row)
        percentage_table.append(percentage_row)
    
    return count_table, percentage_table, tasks


def print_table(table, row_headers, col_headers, title):
    """
    Print a formatted table.
    
    Args:
        table: List of lists representing the table data
        row_headers: List of row header names
        col_headers: List of column header names
        title: Title for the table
    """
    
    print("\n" + "="*120)
    print(f"{title}")
    print("="*120)
    
    # Calculate column widths
    col_widths = [max(len(str(col)), max(len(str(row[i])) for row in table)) + 2 
                  for i, col in enumerate(col_headers)]
    row_header_width = max(len(str(header)) for header in row_headers) + 2
    
    # Print header row
    print(f"{'Format':<{row_header_width}}", end="")
    for i, header in enumerate(col_headers):
        print(f"{header:<{col_widths[i]}}", end="")
    print()
    
    # Print separator
    print("-" * (row_header_width + sum(col_widths)))
    
    # Print data rows
    for i, row in enumerate(table):
        print(f"{row_headers[i]:<{row_header_width}}", end="")
        for j, cell in enumerate(row):
            print(f"{cell:<{col_widths[j]}}", end="")
        print()


def save_results(count_table, percentage_table, row_headers, col_headers, output_dir, results=None):
    """
    Save the analysis results to CSV files.
    
    Args:
        count_table, percentage_table: Table data as lists of lists
        row_headers, col_headers: Headers for rows and columns
        output_dir: Directory to save output files
        results: Optional results dictionary for dataset summary
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save count table
    count_file = os.path.join(output_dir, "benchmark_results_counts_final.csv")
    with open(count_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Format'] + col_headers)
        for i, row in enumerate(count_table):
            writer.writerow([row_headers[i]] + row)
    
    # Save percentage table
    percentage_file = os.path.join(output_dir, "benchmark_results_percentages_final.csv")
    with open(percentage_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Format'] + col_headers)
        for i, row in enumerate(percentage_table):
            writer.writerow([row_headers[i]] + row)
    
    print(f"\nFinal results saved to:")
    print(f"- Count table: {count_file}")
    print(f"- Percentage table: {percentage_file}")


def generate_summary_statistics(results):
    """
    Generate additional summary statistics.
    
    Args:
        results (dict): Results dictionary from analysis
    """
    
    print("\n" + "="*100)
    print("SUMMARY STATISTICS (WITH IMPROVED EVALUATION)")
    print("="*100)
    
    # Overall statistics
    total_questions = sum(r['total'] for r in results.values())
    total_correct = sum(r['correct'] for r in results.values())
    overall_percentage = (total_correct / total_questions * 100) if total_questions > 0 else 0
    
    print(f"Overall Results: {total_correct}/{total_questions} ({overall_percentage:.1f}%)")
    
    # Statistics by task
    task_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for (dataset, task, data_format), result in results.items():
        task_stats[task]['correct'] += result['correct']
        task_stats[task]['total'] += result['total']
    
    print(f"\nResults by Task (Improved Evaluation):")
    for task in sorted(task_stats.keys()):
        stats = task_stats[task]
        pct = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {task}: {stats['correct']}/{stats['total']} ({pct:.1f}%)")


def get_available_models():
    """Get list of available model result directories."""
    results_dir = Path("/insight-fast/dnguyen/Q_Benchmark/benchmark_results")
    if results_dir.exists():
        return sorted([d.name for d in results_dir.iterdir() if d.is_dir()])
    return []


def main():
    """Main function to run the final analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze Q-Benchmark results with improved evaluation (FINAL VERSION)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_analysis_final.py --model gemini-2.5-flash
  python benchmark_analysis_final.py --model gpt-5-mini
  python benchmark_analysis_final.py --list
        """
    )
    
    # Get available options
    available_models = get_available_models()
    
    parser.add_argument("--model", 
                       choices=available_models,
                       default="gemini-2.5-flash",
                       help="Model to analyze (default: gemini-2.5-flash)")
    
    parser.add_argument("--output-dir",
                       default="/insight-fast/dnguyen/Q_Benchmark/analysis_results_final",
                       help="Base output directory for results")
    
    parser.add_argument("--list", action="store_true",
                       help="List available models")
    
    args = parser.parse_args()
    
    # Handle list option
    if args.list:
        print("Available models:")
        for model in available_models:
            print(f"  - {model}")
        return
    
    base_path = f"/insight-fast/dnguyen/Q_Benchmark/benchmark_results/{args.model}"
    output_dir = f"{args.output_dir}/{args.model}"
    
    print("Starting Q-Benchmark results analysis (FINAL VERSION WITH IMPROVED EVALUATION)...")
    print(f"Model: {args.model}")
    print(f"Base path: {base_path}")
    print(f"Output dir: {output_dir}")
    
    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        return
    
    # Run the analysis
    datasets, tasks, data_formats, results = analyze_benchmark_results_final(base_path)
    
    if not results:
        print(f"No results found for {args.model}")
        return
    
    # Create tables
    count_table, percentage_table, task_columns = create_tables(datasets, tasks, data_formats, results)
    
    # Display tables
    print_table(count_table, data_formats, task_columns, 
                f"TABLE 1: CORRECT ANSWERS / TOTAL ANSWERS - {args.model} (IMPROVED EVALUATION)")
    print_table(percentage_table, data_formats, task_columns, 
                f"TABLE 2: PERCENTAGE CORRECT - {args.model} (IMPROVED EVALUATION)")
    
    # Save results
    save_results(count_table, percentage_table, data_formats, task_columns, output_dir, results)
    
    # Generate summary statistics
    generate_summary_statistics(results)
    
    print(f"\nFinal analysis complete for {args.model}!")
    print(f"This analysis uses both robust CSV parsing AND improved evaluation functions.")
    print(f"Results are saved in: {output_dir}")


if __name__ == "__main__":
    main()