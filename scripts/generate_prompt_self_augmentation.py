#!/usr/bin/env python3
"""
Generate converted_prompts_self_aug CSV files matching the original Q-Benchmark approach.

This script generates self-augmented prompts with a [REQUEST] placeholder that gets
substituted at runtime in the benchmark pipeline with different augmentation requests.
"""

import json
import csv
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConvertedPromptsSelfAugGenerator:
    """Generator for converted_prompts_self_aug CSV files matching original approach"""
    
    def __init__(self, advanced_prompts_dir: str, benchmark_cache_dir: str, output_dir: str):
        self.advanced_prompts_dir = Path(advanced_prompts_dir)
        self.benchmark_cache_dir = Path(benchmark_cache_dir)
        self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Format mappings
        self.formats = ['html', 'json', 'md', 'ttl', 'txt', 'xml']

        # Load case metadata
        self._metadata = {}
        metadata_file = Path("case_metadata.csv")
        if metadata_file.exists():
            self._metadata = self._load_metadata(metadata_file)

    def _load_metadata(self, csv_file: Path) -> Dict[str, Dict[str, set]]:
        """Load case metadata"""
        from collections import defaultdict
        metadata = defaultdict(lambda: defaultdict(set))

        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metadata[row['dataset']][row['task']].add(row['case_id'])
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")

        return metadata
        
    def load_advanced_prompts(self, dataset: str, task: str) -> List[Dict[str, Any]]:
        """Load advanced prompts from JSON file"""
        filename = f"{dataset}_{task}_qa_pairs.json"
        filepath = self.advanced_prompts_dir / dataset / filename
        
        if not filepath.exists():
            logger.warning(f"Advanced prompts file not found: {filepath}")
            return []
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} advanced prompts from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading advanced prompts from {filepath}: {e}")
            return []
    
    def load_benchmark_cache(self, dataset: str, task: str, format_type: str, case_id: str) -> Optional[str]:
        """Load benchmark cache data for specific case and format"""
        cache_path = self.benchmark_cache_dir / dataset / task / format_type / f"{case_id}.{format_type}"
        
        if not cache_path.exists():
            logger.warning(f"Cache file not found: {cache_path}")
            return None
            
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"Error loading cache from {cache_path}: {e}")
            return None
    
    def substitute_placeholders(self, prompt_template: str, case_data: Dict[str, Any], questionnaire_data: str, dataset: str, task: str, format_type: str) -> str:
        """Substitute placeholders in prompt template with actual data"""
        try:
            # Replace placeholders with actual data
            filled_prompt = prompt_template
            
            # Handle case data substitution - need to substitute BEFORE replacing case placeholders
            # First create a copy of case_data with questionnaire substituted
            substituted_case_data = {}
            for key, value in case_data.items():
                if isinstance(value, str):
                    # Replace questionnaire placeholders in case data
                    substituted_value = value.replace("[Insert the full data block here]", questionnaire_data)
                    substituted_value = substituted_value.replace("[questionnaire]", questionnaire_data)
                    substituted_case_data[key] = substituted_value
                else:
                    substituted_case_data[key] = value
            
            # Handle case data substitution with substituted values
            for key, value in substituted_case_data.items():
                if key.upper() in filled_prompt:
                    filled_prompt = filled_prompt.replace(f"[{key.upper()}]", str(value))
            
            # Handle questionnaire substitution
            if "[questionnaire]" in filled_prompt:
                filled_prompt = filled_prompt.replace("[questionnaire]", questionnaire_data)
            
            # Handle question substitution
            if "[question]" in filled_prompt and "question" in case_data:
                filled_prompt = filled_prompt.replace("[question]", str(case_data["question"]))
            
            # Handle CASE_1 substitution in example section
            if "[CASE_1]" in filled_prompt:
                case_1_content = self._get_case_1_example(dataset, task, format_type)
                filled_prompt = filled_prompt.replace("[CASE_1]", case_1_content)
            
            # Handle specific case substitution (e.g., [CASE_2], [CASE_3], etc.)
            if "case_id" in case_data:
                case_placeholder = f"[{case_data['case_id'].upper()}]"
                if case_placeholder in filled_prompt and case_placeholder.replace('[', '').replace(']', '') in substituted_case_data:
                    case_content = substituted_case_data[case_placeholder.replace('[', '').replace(']', '')]
                    filled_prompt = filled_prompt.replace(case_placeholder, case_content)
            
            # Final cleanup of any remaining questionnaire placeholders
            filled_prompt = filled_prompt.replace("[Insert the full data block here]", questionnaire_data)
            
            return filled_prompt
            
        except Exception as e:
            logger.error(f"Error substituting placeholders: {e}")
            return prompt_template
    
    def _get_case_1_example(self, dataset: str, task: str, format_type: str) -> str:
        """Get case_1 questionnaire data and expected answer for example section"""
        try:
            # Load case_1 questionnaire data from benchmark cache
            case_1_questionnaire = self.load_benchmark_cache(dataset, task, format_type, "case_1")
            if case_1_questionnaire is None:
                logger.warning(f"Could not load case_1 cache data for {dataset}/{task}/{format_type}")
                return "[CASE_1 data not available]"
            
            # Load case_1 advanced prompts to get expected answer and question
            advanced_prompts = self.load_advanced_prompts(dataset, task)
            case_1_data = None
            for prompt_data in advanced_prompts:
                if prompt_data.get('case_id') == 'case_1':
                    case_1_data = prompt_data
                    break
            
            if case_1_data is None:
                logger.warning(f"Could not find case_1 in advanced prompts for {dataset}/{task}")
                return "[CASE_1 data not available]"
            
            # Format the example: questionnaire + question + expected answer
            case_1_question = case_1_data.get('question', '')
            case_1_expected_answer = case_1_data.get('expected_answer', '')
            
            example_content = f"{case_1_questionnaire}\n\nQuestion: {case_1_question}\nAnswer: {case_1_expected_answer}"
            
            return example_content
            
        except Exception as e:
            logger.error(f"Error generating case_1 example: {e}")
            return "[CASE_1 example generation failed]"
    
    
    def apply_self_aug_transformations(self, prompt: str, self_aug_type: str, question: str = "") -> str:
        """
        Apply improved self-augmentation transformations with clear task prioritization.

        New structure:
        - Task comes first
        - Augmentation is presented as an approach to help answer
        - Clear output format with split symbols
        - Standard calculation methods for critical_values
        """
        import re

        # Extract the task section
        task_match = re.search(r'<task>(.*?)</task>', prompt, re.DOTALL)
        task_content = task_match.group(1).strip() if task_match else ""

        # Remove the old task section - we'll add it back in new location
        processed_prompt = re.sub(r'<task>.*?</task>\s*', '', prompt, flags=re.DOTALL)

        # Remove the old <request> section entirely - we'll replace with new <approach>
        processed_prompt = re.sub(r'<request>.*?</request>\s*', '', processed_prompt, flags=re.DOTALL)

        # Build the new augmentation approach based on type
        if self_aug_type == "format_explaination":
            approach_text = """<approach>
Generate short format specification and description of the survey within five sentences.
</approach>"""
        elif self_aug_type == "critical_values":
            approach_text = """<approach>
Identify critical values and ranges of the survey related within five sentences.
</approach>"""
        elif self_aug_type == "structural_info":
            approach_text = """<approach>
Describe structural information, patterns and statistics of the survey within five sentences.
</approach>"""
        else:
            approach_text = ""

        # Add output format with split symbols (similar style to generate_converted_prompts.py)
        output_format = """
<output_format>
Structure your response:

### ANALYSIS ###
[Your analysis here]

### ANSWER ###
[Your final answer]

### END ###
</output_format>"""

        # Build final prompt: approach first, then output format, then task at the END
        final_prompt = processed_prompt

        # Insert approach and output_format before <output>, then add task at the very end
        output_match = re.search(r'<output>.*?</output>', final_prompt, re.DOTALL)
        if output_match:
            # Insert before <output> section
            insert_pos = output_match.start()
            final_prompt = (final_prompt[:insert_pos] +
                          approach_text + "\n\n" +
                          output_format + "\n\n" +
                          final_prompt[insert_pos:] +
                          f"\n\n<task>\n{task_content}\n</task>")
        else:
            # Append at end
            final_prompt = final_prompt + "\n\n" + approach_text + "\n\n" + output_format + f"\n\n<task>\n{task_content}\n</task>"

        return final_prompt

    def generate_self_aug_csv(self, dataset: str, task: str, format_type: str, self_aug_type: str):
        """Generate converted_prompts_self_aug CSV for specific format and self-augmentation type"""
        logger.info(f"Generating converted_prompts_self_aug for {dataset}/{task}/{format_type} with {self_aug_type}")
        
        # Load advanced prompts
        advanced_prompts = self.load_advanced_prompts(dataset, task)
        if not advanced_prompts:
            return
        
        # Prepare output with self_aug_type subdirectory
        output_path = self.output_dir / self_aug_type / dataset / task
        output_path.mkdir(parents=True, exist_ok=True)
        
        csv_filename = f"{task}_{format_type}_converted_prompts.csv"
        csv_path = output_path / csv_filename
        
        # CSV headers (same as basic converted_prompts)
        headers = ['case_id', 'task', 'question', 'questionnaire', 'expected_answer', 'prompt', 'Response', 'Correct', 'skip']
        
        rows = []
        
        for prompt_data in advanced_prompts:
            case_id = prompt_data.get('case_id', '')
            
            # Load questionnaire data from cache
            questionnaire_data = self.load_benchmark_cache(dataset, task, format_type, case_id)
            if questionnaire_data is None:
                logger.warning(f"Skipping {case_id} - no cache data found for {format_type}")
                continue
            
            # Generate the base prompt by substituting placeholders
            prompt_template = prompt_data.get('prompt', '')
            base_prompt = self.substitute_placeholders(prompt_template, prompt_data, questionnaire_data, dataset, task, format_type)
            
            # Apply complete self-augmentation transformations
            question = prompt_data.get('question', '')
            self_aug_prompt = self.apply_self_aug_transformations(base_prompt, self_aug_type, question)

            # Set metadata flag
            skip_flag = 'FALSE'
            if dataset in self._metadata and task in self._metadata[dataset]:
                if case_id in self._metadata[dataset][task]:
                    skip_flag = 'TRUE'

            # Create CSV row
            row = {
                'case_id': case_id,
                'task': task,
                'question': prompt_data.get('question', ''),
                'questionnaire': questionnaire_data,
                'expected_answer': prompt_data.get('expected_answer', ''),
                'prompt': self_aug_prompt,
                'Response': '',  # Empty for new generation
                'Correct': '',    # Empty for new generation
                'skip': skip_flag
            }
            
            rows.append(row)
        
        # Write CSV file
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info(f"Generated {csv_path} with {len(rows)} rows")
            
        except Exception as e:
            logger.error(f"Error writing CSV to {csv_path}: {e}")
    
    def generate_all_self_aug_prompts(self, self_aug_types: List[str] = None):
        """Generate all converted_prompts_self_aug CSV files for specified self-augmentation types"""
        if self_aug_types is None:
            self_aug_types = ["format_explaination", "critical_values", "structural_info"]
        
        logger.info(f"Starting generation of all converted_prompts_self_aug CSV files for: {self_aug_types}")
        
        # Get all datasets
        datasets = [d.name for d in self.advanced_prompts_dir.iterdir() if d.is_dir()]
        
        for self_aug_type in self_aug_types:
            logger.info(f"Processing self-augmentation type: {self_aug_type}")
            
            for dataset in datasets:
                logger.info(f"Processing dataset: {dataset}")
                
                # Get all tasks for this dataset
                dataset_path = self.advanced_prompts_dir / dataset
                task_files = list(dataset_path.glob("*_qa_pairs.json"))
                
                tasks = []
                for task_file in task_files:
                    # Extract task name from filename
                    filename = task_file.stem
                    task_name = filename.replace(f"{dataset}_", "").replace("_qa_pairs", "")
                    tasks.append(task_name)
                
                for task in tasks:
                    logger.info(f"Processing task: {task}")
                    
                    for format_type in self.formats:
                        self.generate_self_aug_csv(dataset, task, format_type, self_aug_type)
        
        logger.info("Completed generation of all converted_prompts_self_aug CSV files")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate converted_prompts_self_aug CSV files")
    parser.add_argument("--advanced-prompts", default="advanced_prompts", 
                       help="Path to advanced_prompts directory")
    parser.add_argument("--benchmark-cache", default="benchmark_cache",
                       help="Path to benchmark_cache directory") 
    parser.add_argument("--output", default="converted_prompts_self_aug",
                       help="Output directory for generated CSV files")
    parser.add_argument("--self-aug-types", nargs='+', 
                       choices=["format_explaination", "critical_values", "structural_info"],
                       help="Self-augmentation types to generate (default: all)")
    
    args = parser.parse_args()
    
    generator = ConvertedPromptsSelfAugGenerator(
        advanced_prompts_dir=args.advanced_prompts,
        benchmark_cache_dir=args.benchmark_cache,
        output_dir=args.output
    )
    
    generator.generate_all_self_aug_prompts(args.self_aug_types)


if __name__ == "__main__":
    main()