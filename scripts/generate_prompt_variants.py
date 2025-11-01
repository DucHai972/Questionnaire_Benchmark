#!/usr/bin/env python3
"""
Generate converted_prompts_variants CSV files matching the original Q-Benchmark approach.

This script generates ablation study variants by selectively removing or modifying 
prompt sections to test their impact on model performance.

Variants generated:
- wo_change_order: Preserves original section order (doesn't move questionnaire first)
- wo_format_explaination: Removes the <format> section
- wo_oneshot: Removes the <example> section  
- wo_partition_mark: Removes all section markers
- wo_role_prompting: Removes the <role> section
"""

import json
import csv
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConvertedPromptsVariantsGenerator:
    """Generator for converted_prompts_variants CSV files using ablation studies"""
    
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

        # Variant configurations
        self.variants = {
            'wo_change_order': {
                'description': 'Moves questionnaire section to the end of prompt',
                'modifications': ['preserve_order']
            },
            'wo_format_explaination': {
                'description': 'Removes format explanation section',
                'modifications': ['remove_format']
            },
            'wo_oneshot': {
                'description': 'Removes example section',
                'modifications': ['remove_example']
            },
            'wo_partition_mark': {
                'description': 'Removes all section markers',
                'modifications': ['remove_all_sections']
            },
            'wo_role_prompting': {
                'description': 'Removes role prompting section',
                'modifications': ['remove_role']
            }
        }

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
    
    def apply_variant_modifications(self, prompt: str, variant_name: str) -> str:
        """Apply variant-specific modifications to the prompt"""
        variant_config = self.variants[variant_name]
        modifications = variant_config['modifications']
        
        modified_prompt = prompt
        
        for modification in modifications:
            if modification == 'preserve_order':
                # wo_change_order: Keep original section order, don't move questionnaire first
                # This means we use the original template order, not the reordered one
                # In the basic version, questionnaire is moved to after example
                # In wo_change_order, we keep the original order: example -> questionnaire -> role -> format -> request -> output -> task
                modified_prompt = self._preserve_original_order(modified_prompt)
                
            elif modification == 'remove_format':
                # wo_format_explaination: Remove <format> section
                modified_prompt = re.sub(r'<format>.*?</format>\s*', '', modified_prompt, flags=re.DOTALL)
                
            elif modification == 'remove_example':
                # wo_oneshot: Remove <example> section
                modified_prompt = re.sub(r'<example>.*?</example>\s*', '', modified_prompt, flags=re.DOTALL)
                
            elif modification == 'remove_all_sections':
                # wo_partition_mark: Remove all section markers but keep content
                # Remove opening tags
                modified_prompt = re.sub(r'<(example|questionnaire|role|format|request|output|task)>\s*', '', modified_prompt)
                # Remove closing tags
                modified_prompt = re.sub(r'\s*</(example|questionnaire|role|format|request|output|task)>', '', modified_prompt)
                # Clean up extra whitespace
                modified_prompt = re.sub(r'\n\n\n+', '\n\n', modified_prompt)
                
            elif modification == 'remove_role':
                # wo_role_prompting: Remove <role> section
                modified_prompt = re.sub(r'<role>.*?</role>\s*', '', modified_prompt, flags=re.DOTALL)
        
        return modified_prompt.strip()
    
    def _preserve_original_order(self, prompt: str) -> str:
        """Move questionnaire to the end for wo_change_order variant"""
        # The basic prompt has questionnaire early in the sequence
        # wo_change_order moves questionnaire to the very end
        
        # Extract all sections
        sections = {}
        
        # Extract each section with regex
        section_patterns = {
            'example': r'(<example>.*?</example>)',
            'questionnaire': r'(<questionnaire>.*?</questionnaire>)',
            'role': r'(<role>.*?</role>)',
            'format': r'(<format>.*?</format>)',
            'request': r'(<request>.*?</request>)',
            'output': r'(<output>.*?</output>)',
            'task': r'(<task>.*?</task>)'
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, prompt, re.DOTALL)
            if match:
                sections[section_name] = match.group(1)
        
        # Rebuild with questionnaire moved to the end: example -> role -> format -> request -> output -> task -> questionnaire
        ordered_sections = []
        for section_name in ['example', 'role', 'format', 'request', 'output', 'task', 'questionnaire']:
            if section_name in sections:
                ordered_sections.append(sections[section_name])
        
        return '\n\n'.join(ordered_sections)
    
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
    
    def generate_variant_csv(self, dataset: str, task: str, format_type: str, variant_name: str):
        """Generate CSV for a specific variant"""
        logger.info(f"Generating {variant_name} variant for {dataset}/{task}/{format_type}")
        
        # Load advanced prompts
        advanced_prompts = self.load_advanced_prompts(dataset, task)
        if not advanced_prompts:
            return
        
        # Prepare output directory
        variant_output_dir = self.output_dir / variant_name / dataset / task
        variant_output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_filename = f"{task}_{format_type}_converted_prompts.csv"
        csv_path = variant_output_dir / csv_filename
        
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
            
            # Generate the basic prompt by substituting placeholders
            prompt_template = prompt_data.get('prompt', '')
            base_prompt = self.substitute_placeholders(prompt_template, prompt_data, questionnaire_data, dataset, task, format_type)
            
            # Apply variant-specific modifications
            variant_prompt = self.apply_variant_modifications(base_prompt, variant_name)

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
                'prompt': variant_prompt,
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
    
    def generate_all_variants(self):
        """Generate all variant CSV files"""
        logger.info("Starting generation of all converted_prompts_variants CSV files")
        
        # Get all datasets
        datasets = [d.name for d in self.advanced_prompts_dir.iterdir() if d.is_dir()]
        
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
                    for variant_name in self.variants.keys():
                        self.generate_variant_csv(dataset, task, format_type, variant_name)
        
        logger.info("Completed generation of all converted_prompts_variants CSV files")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate converted_prompts_variants CSV files")
    parser.add_argument("--advanced-prompts", default="advanced_prompts", 
                       help="Path to advanced_prompts directory")
    parser.add_argument("--benchmark-cache", default="benchmark_cache",
                       help="Path to benchmark_cache directory") 
    parser.add_argument("--output", default="converted_prompts_variants",
                       help="Output directory for generated CSV files")
    
    args = parser.parse_args()
    
    generator = ConvertedPromptsVariantsGenerator(
        advanced_prompts_dir=args.advanced_prompts,
        benchmark_cache_dir=args.benchmark_cache,
        output_dir=args.output
    )
    
    generator.generate_all_variants()


if __name__ == "__main__":
    main()