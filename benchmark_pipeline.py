#!/usr/bin/env python3
"""
Simple Benchmark Pipeline for Q-Benchmark

This pipeline reads prompts from converted_prompts CSV files, runs them through
OpenAI and Google Gemini models, evaluates responses, and updates CSV files
with results.
"""

import os
import csv
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
# Progress bar - use tqdm if available, otherwise simple counter
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleLLMClient:
    """Base class for LLM API clients"""
    
    def __init__(self, api_key: str, model_name: str, provider: str):
        self.api_key = api_key
        self.model_name = model_name
        self.provider = provider
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate response from LLM"""
        raise NotImplementedError


class SimpleOpenAIClient(SimpleLLMClient):
    """Simple OpenAI API Client"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4.1-mini"):
        super().__init__(api_key, model_name, "openai")
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate response using OpenAI API"""
        start_time = time.time()
        try:
            # Use max_completion_tokens for newer models (GPT-4, GPT-5, etc.)
            # and max_tokens for legacy models
            # Some models don't support temperature=0, so use default for those
            if self.model_name.startswith(('gpt-4', 'gpt-5', 'o1-')):
                # o1 models don't support temperature parameter at all
                if self.model_name.startswith('o1-'):
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=max_tokens
                    )
                else:
                    # GPT-4, GPT-5 models - some may not support temperature=0
                    # GPT-5-mini specifically doesn't support temperature=0
                    if self.model_name in ['gpt-5-mini', 'gpt-5-mini-2025-08-07']:
                        # Use default temperature for GPT-5-mini
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            max_completion_tokens=max_tokens
                        )
                    else:
                        try:
                            response = self.client.chat.completions.create(
                                model=self.model_name,
                                messages=[{"role": "user", "content": prompt}],
                                max_completion_tokens=max_tokens,
                                temperature=0
                            )
                        except Exception as e:
                            if "temperature" in str(e):
                                # Retry without temperature parameter
                                response = self.client.chat.completions.create(
                                    model=self.model_name,
                                    messages=[{"role": "user", "content": prompt}],
                                    max_completion_tokens=max_tokens
                                )
                            else:
                                raise
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0
                )
            
            end_time = time.time()
            response_content = response.choices[0].message.content or ""
            
            return {
                "response": response_content.strip(),
                "success": True,
                "error": None,
                "response_time": end_time - start_time
            }
        except Exception as e:
            end_time = time.time()
            logger.error(f"OpenAI API error: {e}")
            return {
                "response": "",
                "success": False,
                "error": str(e),
                "response_time": end_time - start_time
            }


class SimpleGoogleClient(SimpleLLMClient):
    """Simple Google Gemini API Client"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        super().__init__(api_key, model_name, "google")
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
    
    def generate(self, prompt: str, max_tokens: int = 8192) -> Dict[str, Any]:
        """Generate response using Google Gemini API"""
        start_time = time.time()
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0,
                    "top_p": 1
                }
            )
            
            end_time = time.time()
            response_text = response.text if response.text else ""
            
            return {
                "response": response_text.strip(),
                "success": True,
                "error": None,
                "response_time": end_time - start_time
            }
        except Exception as e:
            end_time = time.time()
            logger.error(f"Google API error: {e}")
            return {
                "response": "",
                "success": False,
                "error": str(e),
                "response_time": end_time - start_time
            }




class SimpleBenchmarkPipeline:
    """Simple benchmark pipeline for processing CSV prompt files"""
    
    def __init__(self, converted_prompts_dir: str = "converted_prompts", init_clients: bool = True, 
                 openai_model: str = "gpt-3.5-turbo", google_model: str = "gemini-1.5-flash", 
                 variants: str = None):
        load_dotenv()
        
        self.converted_prompts_dir = Path(converted_prompts_dir)
        self.results_dir = Path("benchmark_results")
        self.openai_model = openai_model
        self.google_model = google_model
        self.variants = variants
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize LLM clients only if requested
        if init_clients:
            self.clients = self._initialize_clients()
            logger.info(f"Initialized pipeline with {len(self.clients)} LLM clients")
        else:
            self.clients = {}
    
    def _initialize_clients(self) -> Dict[str, SimpleLLMClient]:
        """Initialize LLM clients from .env file"""
        clients = {}
        
        # Initialize OpenAI client
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                clients["openai"] = SimpleOpenAIClient(openai_key, self.openai_model)
                logger.info(f"Initialized OpenAI client with model: {self.openai_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning("OPENAI_API_KEY not found in .env file")
        
        # Initialize Google client
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            try:
                clients["google"] = SimpleGoogleClient(google_key, self.google_model)
                logger.info(f"Initialized Google client with model: {self.google_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Google client: {e}")
        else:
            logger.warning("GOOGLE_API_KEY not found in .env file")
        
        if not clients:
            raise ValueError("No valid clients initialized. Please add API keys to .env file")
        
        return clients
    
    def load_csv_prompts(self, csv_file: Path) -> List[Dict[str, str]]:
        """Load prompts from CSV file with corruption detection and recovery"""
        prompts = []
        try:
            with open(csv_file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                corrupted_rows = 0
                for row in reader:
                    # Check for corruption (case_id not starting with 'case_')
                    case_id = row.get('case_id', '').strip()
                    if case_id and not case_id.startswith('case_'):
                        corrupted_rows += 1
                        continue
                    prompts.append(dict(row))
                
                if corrupted_rows > 0:
                    logger.warning(f"Found {corrupted_rows} corrupted rows in {csv_file}, attempting perfect recovery")
                    return self.perfect_recovery_csv(csv_file)
                    
            return prompts
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_file}: {e}")
            logger.info(f"Attempting perfect recovery for {csv_file}")
            return self.perfect_recovery_csv(csv_file)
    
    def perfect_recovery_csv(self, benchmark_file: Path) -> List[Dict[str, str]]:
        """Perform perfect recovery by mapping to converted_prompts source"""
        try:
            # Find corresponding source file in converted_prompts
            source_file = self.find_source_file_for_recovery(benchmark_file)
            if not source_file or not source_file.exists():
                logger.error(f"Source file not found for recovery: {source_file}")
                return []
            
            # Load source data (authoritative)
            source_data = []
            with open(source_file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    source_data.append(dict(row))
            
            # Load benchmark data (potentially corrupted) for response merging
            benchmark_data = []
            benchmark_lookup = {}
            try:
                with open(benchmark_file, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        benchmark_data.append(dict(row))
                        case_id = row.get('case_id', '').strip()
                        # Only trust valid case_ids for response data
                        if case_id and case_id.startswith('case_'):
                            benchmark_lookup[case_id] = row
            except:
                pass  # If benchmark file is completely corrupted, use empty lookup
            
            # Perform perfect recovery
            recovered_data = []
            for source_row in source_data:
                case_id = source_row.get('case_id', '').strip()
                recovered_row = source_row.copy()
                
                # Merge response data from benchmark if available and valid
                if case_id in benchmark_lookup:
                    benchmark_row = benchmark_lookup[case_id]
                    for field in ['Response']:
                        if field in benchmark_row and benchmark_row[field]:
                            recovered_row[field] = benchmark_row[field]
                        elif field not in recovered_row:
                            recovered_row[field] = ''
                else:
                    # This case was missing or corrupted in benchmark
                    for field in ['Response']:
                        if field not in recovered_row:
                            recovered_row[field] = ''
                
                recovered_data.append(recovered_row)
            
            corrupted_count = len(benchmark_data) - len(benchmark_lookup)
            logger.info(f"Perfect recovery completed: {len(recovered_data)} rows recovered, {corrupted_count} corrupted rows fixed")
            return recovered_data
            
        except Exception as e:
            logger.error(f"Perfect recovery failed for {benchmark_file}: {e}")
            return []
    
    def determine_source_directory(self, model_name: str) -> tuple[Path, Optional[str]]:
        """
        Determine the correct source directory based on model name patterns.
        
        Returns:
            tuple: (source_directory_path, subdirectory_or_none)
        """
        base_dir = Path.cwd()
        
        # Self-augmentation models: *_critical_values, *_format_explaination, *_structural_info
        self_aug_patterns = ["critical_values", "format_explaination", "structural_info"]
        for pattern in self_aug_patterns:
            if model_name.endswith(f"_{pattern}"):
                return base_dir / "converted_prompts_self_aug", pattern
        
        # Variant models: *_wo_*
        variant_patterns = ["wo_role_prompting", "wo_partition_mark", "wo_format_explaination", 
                           "wo_oneshot", "wo_change_order"]
        for pattern in variant_patterns:
            if model_name.endswith(f"_{pattern}"):
                return base_dir / "converted_prompts_variants", pattern
        
        # Base models: gpt-5-mini, gemini-2.5-flash (exact matches or base models)
        base_models = ["gpt-5-mini", "gemini-2.5-flash", "gpt-4o-mini", "gpt-4", "gemini-1.5-flash"]
        if model_name in base_models:
            return base_dir / "converted_prompts", None
            
        # Handle other base model variations (e.g., gpt-4.1-mini, etc.)
        # If it doesn't match any pattern above, it's likely a base model
        logger.info(f"Model '{model_name}' doesn't match known patterns, assuming base model")
        return base_dir / "converted_prompts", None

    def find_source_file_for_recovery(self, benchmark_file: Path) -> Optional[Path]:
        """Find corresponding source file with intelligent directory mapping"""
        try:
            # For benchmark_results files, map back to appropriate source directory
            if "benchmark_results" in str(benchmark_file):
                # Extract relative path from benchmark_results
                benchmark_results_path = self.results_dir
                relative_path = benchmark_file.relative_to(benchmark_results_path)
                
                # Extract model name and path components
                model_name = relative_path.parts[0]  # First part is model directory
                path_parts = relative_path.parts[1:]  # Rest is dataset/task/filename
                
                # Determine source directory using intelligent mapping
                source_dir, subdirectory = self.determine_source_directory(model_name)
                
                # Build source file path
                if subdirectory:
                    # For self-aug and variants: source_dir/subdirectory/dataset/task/filename
                    source_file = source_dir / subdirectory / Path(*path_parts)
                else:
                    # For base models: source_dir/dataset/task/filename
                    source_file = source_dir / Path(*path_parts)
                
                logger.debug(f"Model '{model_name}' maps to source: {source_file}")
                return source_file
            else:
                # For converted_prompts files being processed directly, use as-is
                return benchmark_file
        except Exception as e:
            logger.error(f"Error finding source file for {benchmark_file}: {e}")
            return None
    
    def save_csv_results(self, csv_file: Path, prompts: List[Dict[str, str]], provider_name: str, self_aug_type: Optional[str] = None) -> bool:
        """Save prompts results to benchmark_results directory organized by model with smart merging"""
        try:
            # Get actual model name from client
            client = self.clients[provider_name]
            actual_model_name = client.model_name
            # Clean up model name for directory (remove slashes)
            actual_model_name = actual_model_name.replace("/", "-").replace("\\", "-")
            
            # Create model-specific directory using actual model name
            if self_aug_type:
                model_dir_name = f"{actual_model_name}_{self_aug_type}"
            elif self.variants:
                model_dir_name = f"{actual_model_name}_{self.variants}"
            else:
                model_dir_name = actual_model_name
            model_dir = self.results_dir / model_dir_name
            model_dir.mkdir(exist_ok=True)
            
            # Recreate the same structure as converted_prompts
            relative_path = csv_file.relative_to(self.converted_prompts_dir)
            output_file = model_dir / relative_path
            
            # Ensure parent directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing CSV data if file exists (smart merging with corruption handling)
            existing_data = {}
            if output_file.exists():
                try:
                    corrupted_rows = 0
                    with open(output_file, 'r', newline='', encoding='utf-8') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            case_id = row.get('case_id', '').strip()
                            # Check for corruption
                            if case_id and not case_id.startswith('case_'):
                                corrupted_rows += 1
                                continue
                            existing_data[row['case_id']] = row
                    
                    if corrupted_rows > 0:
                        logger.warning(f"Found {corrupted_rows} corrupted rows in existing file {output_file}, using perfect recovery")
                        # Load with perfect recovery
                        recovered_data = self.perfect_recovery_csv(output_file)
                        existing_data = {}
                        for row in recovered_data:
                            existing_data[row['case_id']] = row
                        
                except Exception as e:
                    logger.error(f"Error loading existing file {output_file}: {e}")
                    logger.info(f"Attempting perfect recovery for existing file")
                    # Try perfect recovery on corrupted existing file
                    recovered_data = self.perfect_recovery_csv(output_file)
                    existing_data = {}
                    for row in recovered_data:
                        existing_data[row['case_id']] = row
            
            # Update existing data with new prompts (overwrite or add)
            for prompt in prompts:
                case_id = prompt['case_id']
                if case_id in existing_data:
                    logger.info(f"Updating existing case_id: {case_id}")
                else:
                    logger.info(f"Adding new case_id: {case_id}")
                existing_data[case_id] = prompt
            
            # Sort by case_id (natural sorting for case_1, case_2, etc.)
            def natural_sort_key(case_id):
                # Extract number from case_id (e.g., "case_1" -> 1)
                try:
                    if case_id.startswith('case_'):
                        return int(case_id.split('_')[1])
                    else:
                        return float('inf')  # Put non-standard case_ids at the end
                except (ValueError, IndexError):
                    return float('inf')
            
            sorted_prompts = sorted(existing_data.values(), key=lambda x: natural_sort_key(x['case_id']))
            
            fieldnames = ["case_id", "task", "question", "questionnaire", 
                         "expected_answer", "prompt", "Response"]
            
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for prompt in sorted_prompts:
                    writer.writerow(prompt)
            return True
        except Exception as e:
            logger.error(f"Error saving results file: {e}")
            return False
    
    def process_csv_file(self, csv_file: Path, model: str, max_cases: Optional[int] = None, start_case: int = 2, overall_pbar=None, self_aug_type: Optional[str] = None) -> bool:
        """Process a single CSV file with specified model"""
        logger.info(f"Processing {csv_file} with {model}")
        
        # Load prompts
        prompts = self.load_csv_prompts(csv_file)
        if not prompts:
            logger.warning(f"No prompts found in {csv_file}")
            return False
        
        # Apply start_case filtering (convert to 0-based index)
        if start_case > 1:
            prompts = prompts[start_case-1:]
            logger.info(f"Starting from case {start_case}, processing {len(prompts)} remaining cases")
        
        # Limit cases if specified
        if max_cases:
            prompts = prompts[:max_cases]
        
        # Get client
        if model not in self.clients:
            logger.error(f"Model {model} not available. Available models: {list(self.clients.keys())}")
            return False
        
        client = self.clients[model]
        
        # Process each prompt
        processed_count = 0
        
        for prompt_data in prompts:
            # Always process (don't skip based on existing Response field)
            
            # For self-aug prompts, they are already fully processed during generation
            # No runtime processing needed - use prompt as-is
            processed_prompt = prompt_data["prompt"]
            
            # Generate response
            result = client.generate(processed_prompt)
            
            # Check if blocked by safety filter
            if not result["success"] and "finish_reason" in str(result.get("error", "")) and "2" in str(result.get("error", "")):
                logger.warning(f"🛡️ Safety filter detected for case_id {prompt_data.get('case_id', 'unknown')}, inserting safety filter response")
                
                # Insert safety filter blocked response
                prompt_data["Response"] = "BLOCKED_BY_SAFETY_FILTER"
                logger.info(f"🛡️ Inserted safety filter response for case_id {prompt_data.get('case_id', 'unknown')}")
            else:
                # Update prompt data with actual response (successful or failed)
                prompt_data["Response"] = result["response"]
            
            # No evaluation - just store the response
            
            processed_count += 1
            
            # Update overall progress bar
            if overall_pbar:
                overall_pbar.update(1)
            elif processed_count % 5 == 0:  # Log progress every 5 prompts
                logger.info(f"Processed {processed_count}/{len(prompts)} prompts")
            
            # Rate limiting
            time.sleep(0.5)
        
        # Save results to benchmark_results directory
        success = self.save_csv_results(csv_file, prompts, model, self_aug_type)
        if success:
            actual_model_name = self.clients[model].model_name
            logger.info(f"Saved results for {actual_model_name} with {processed_count} responses")
        return success
    
    def find_csv_files(self, dataset: Optional[str] = None, 
                       task: Optional[str] = None, 
                       format_type: Optional[str] = None) -> List[Path]:
        """Find CSV files matching criteria"""
        csv_files = []
        
        if not self.converted_prompts_dir.exists():
            logger.error(f"Converted prompts directory not found: {self.converted_prompts_dir}")
            return []
        
        # Build search pattern
        pattern_parts = []
        
        if dataset:
            pattern_parts.append(dataset)
        else:
            pattern_parts.append("*")
        
        if task:
            pattern_parts.append(task)
        else:
            pattern_parts.append("*")
        
        # Search for CSV files
        search_pattern = "/".join(pattern_parts) + "/*.csv"
        
        for csv_file in self.converted_prompts_dir.glob(search_pattern):
            # Filter by format if specified (case-insensitive)
            if format_type and format_type.lower() not in csv_file.stem.lower():
                continue
            csv_files.append(csv_file)
        
        return sorted(csv_files)
    
    def run_benchmark(self, 
                     dataset: Optional[str] = None,
                     task: Optional[str] = None,
                     format_type: Optional[str] = None,
                     model: Optional[str] = None,
                     max_cases: Optional[int] = None,
                     start_case: int = 2,
                     self_aug_type: Optional[str] = None) -> bool:
        """Run benchmark on specified criteria"""
        
        # Find CSV files to process
        csv_files = self.find_csv_files(dataset, task, format_type)
        if not csv_files:
            logger.warning("No CSV files found matching criteria")
            return False
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        # Print discovered files
        print(f"DISCOVERED FILES ({len(csv_files)}):")
        for i, csv_file in enumerate(csv_files, 1):
            rel_path = csv_file.relative_to(self.converted_prompts_dir)
            print(f"  {i}. {rel_path}")
        print()
        
        # Determine models to use
        models_to_use = []
        if model and model in self.clients:
            models_to_use = [model]
        elif model:
            logger.error(f"Model {model} not available. Available: {list(self.clients.keys())}")
            return False
        else:
            models_to_use = list(self.clients.keys())
        
        # Print models that will be used
        print(f"MODELS TO USE: {models_to_use}")
        print()
        
        # Calculate total operations for progress bar
        total_operations = 0
        for csv_file in csv_files:
            prompts = self.load_csv_prompts(csv_file)
            # Apply start_case and max_cases filtering
            if start_case > 1:
                prompts = prompts[start_case-1:]  # Convert to 0-based index
            if max_cases:
                prompts = prompts[:max_cases]
            total_operations += len(prompts) * len(models_to_use)
        
        # Create overall progress bar
        if HAS_TQDM:
            overall_pbar = tqdm(total=total_operations, desc="Overall Progress")
        else:
            overall_pbar = None
            logger.info(f"Starting benchmark: {total_operations} total operations")
        
        # Process each CSV file with each model
        success_count = 0
        total_count = len(csv_files) * len(models_to_use)
        
        try:
            for csv_file in csv_files:
                for model_name in models_to_use:
                    if self.process_csv_file(csv_file, model_name, max_cases, start_case, overall_pbar, self_aug_type):
                        success_count += 1
        finally:
            if overall_pbar:
                overall_pbar.close()
        
        logger.info(f"Benchmark completed: {success_count}/{total_count} files processed successfully")
        
        # Store actual model names used for summary
        self.models_used = [self.clients[model].model_name for model in models_to_use]
        return success_count == total_count


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Simple benchmark pipeline for Q-Benchmark converted prompts")
    
    parser.add_argument("--dataset", help="Dataset to process (default: all)")
    parser.add_argument("--task", help="Task to process (default: all)")
    parser.add_argument("--format", help="Data format to process (default: all)")
    parser.add_argument("--variants", 
                       choices=["all", "wo_role_prompting", "wo_partition_mark", "wo_format_explaination", 
                               "wo_oneshot", "wo_change_order"],
                       help="Prompt variant to use instead of standard prompts. Use 'all' to run all variants. Available variants: wo_role_prompting, wo_partition_mark, wo_format_explaination, wo_oneshot, wo_change_order")
    parser.add_argument("--self_aug", 
                       choices=["format_explaination", "critical_values", "structural_info"],
                       help="Use self-augmentation prompts with specific request type. Mutually exclusive with --variants.")
    parser.add_argument("--model", choices=["openai", "google"], 
                       help="Model provider to use (default: all available)")
    parser.add_argument("--openai-model", default="gpt-3.5-turbo",
                       help="OpenAI model name (default: gpt-3.5-turbo). Examples: gpt-4o-mini, gpt-4, gpt-3.5-turbo")
    parser.add_argument("--google-model", default="gemini-1.5-flash", 
                       help="Google model name (default: gemini-1.5-flash)")
    parser.add_argument("--max-cases", type=int, 
                       help="Maximum cases to process per file")
    parser.add_argument("--start-case", type=int, default=2,
                       help="Starting case number (default: 2)")
    parser.add_argument("--converted-prompts-dir", default="converted_prompts",
                       help="Directory containing converted prompt CSV files")
    parser.add_argument("--list", action="store_true",
                       help="List available datasets, tasks, and formats")
    
    args = parser.parse_args()
    
    # Validate mutually exclusive arguments
    if args.variants and args.self_aug:
        parser.error("--variants and --self_aug are mutually exclusive. Use one or the other.")
    
    # Define REQUEST messages for self_aug types
    SELF_AUG_REQUESTS = {
        "format_explaination": "Generate short format specification and description of the survey within five sentences.",
        "critical_values": "Identify critical values and ranges of the survey related within five sentences.",
        "structural_info": "Describe structural information, patterns and statistics of the survey within five sentences."
    }
    
    # Print configuration settings
    if not args.list:
        print("="*60)
        print("BENCHMARK CONFIGURATION")
        print("="*60)
        print(f"Dataset:              {args.dataset or 'ALL'}")
        print(f"Task:                 {args.task or 'ALL'}")
        print(f"Format:               {args.format or 'ALL'}")
        if args.self_aug:
            print(f"Self-Augmentation:    {args.self_aug}")
        else:
            print(f"Variants:             {args.variants or 'STANDARD'}")
        
        # Only print the model we're actually using
        if args.model == "openai":
            print(f"Model:                {args.openai_model}")
        elif args.model == "google":
            print(f"Model:                {args.google_model}")
        else:
            print(f"OpenAI Model:         {args.openai_model}")
            print(f"Google Model:         {args.google_model}")
            
        print(f"Max Cases per File:   {args.max_cases or 'UNLIMITED'}")
        print(f"Starting Case:        {args.start_case}")
        if args.self_aug:
            print(f"Prompts Directory:    converted_prompts_self_aug/{args.self_aug}")
        elif args.variants:
            print(f"Prompts Directory:    converted_prompts_variants/{args.variants}")
        else:
            print(f"Prompts Directory:    {args.converted_prompts_dir}")
        print("="*60)
        print()

    # Handle --variants all case
    if args.variants == "all":
        available_variants = ["wo_role_prompting", "wo_partition_mark", "wo_format_explaination", 
                             "wo_oneshot", "wo_change_order"]
        
        print(f"\n{'='*60}")
        print(f"RUNNING ALL VARIANTS: {len(available_variants)} variants + standard")
        print(f"{'='*60}")
        
        all_results = []
        
        # Run standard version first (no variants)
        print(f"\n[1/{len(available_variants)+1}] Running STANDARD (no variants)")
        try:
            pipeline = SimpleBenchmarkPipeline(
                args.converted_prompts_dir, 
                init_clients=True,
                openai_model=args.openai_model,
                google_model=args.google_model,
                variants=None
            )
            success = pipeline.run_benchmark(
                dataset=args.dataset,
                task=args.task,
                format_type=args.format,
                model=args.model,
                max_cases=args.max_cases,
                start_case=args.start_case
            )
            all_results.append(("STANDARD", success))
            print(f"STANDARD: {'SUCCESS' if success else 'FAILED'}")
        except Exception as e:
            logger.error(f"Failed to run standard version: {e}")
            all_results.append(("STANDARD", False))
        
        # Run each variant
        for i, variant in enumerate(available_variants, 2):
            print(f"\n[{i}/{len(available_variants)+1}] Running variant: {variant}")
            try:
                prompts_dir = f"converted_prompts_variants/{variant}"
                pipeline = SimpleBenchmarkPipeline(
                    prompts_dir, 
                    init_clients=True,
                    openai_model=args.openai_model,
                    google_model=args.google_model,
                    variants=variant
                )
                success = pipeline.run_benchmark(
                    dataset=args.dataset,
                    task=args.task,
                    format_type=args.format,
                    model=args.model,
                    max_cases=args.max_cases,
                    start_case=args.start_case
                )
                all_results.append((variant, success))
                print(f"{variant}: {'SUCCESS' if success else 'FAILED'}")
            except Exception as e:
                logger.error(f"Failed to run variant {variant}: {e}")
                all_results.append((variant, False))
        
        # Print final summary for all variants
        print(f"\n{'='*80}")
        print("ALL VARIANTS BENCHMARK COMPLETED")
        print(f"{'='*80}")
        success_count = sum(1 for _, success in all_results if success)
        total_count = len(all_results)
        print(f"Overall Status: {success_count}/{total_count} variants completed successfully")
        print(f"Results:")
        for variant_name, success in all_results:
            status = "SUCCESS" if success else "FAILED"
            print(f"  - {variant_name:<25}: {status}")
        print(f"Results Saved To: benchmark_results/ (check individual model directories)")
        print(f"{'='*80}")
        
        return 0 if success_count == total_count else 1
    
    # Initialize pipeline for single variant or standard
    try:
        # Determine the prompts directory based on variants or self_aug argument
        if args.self_aug:
            prompts_dir = f"converted_prompts_self_aug/{args.self_aug}"
        elif args.variants:
            prompts_dir = f"converted_prompts_variants/{args.variants}"
        else:
            prompts_dir = args.converted_prompts_dir
            
        # Don't initialize clients if just listing
        pipeline = SimpleBenchmarkPipeline(
            prompts_dir, 
            init_clients=not args.list,
            openai_model=args.openai_model,
            google_model=args.google_model,
            variants=args.variants
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return 1
    
    # List available options
    if args.list:
        if args.self_aug:
            print(f"Available options in converted_prompts_self_aug/{args.self_aug} directory:")
        elif args.variants:
            print(f"Available options in converted_prompts_variants/{args.variants} directory:")
        else:
            print("Available options in converted_prompts directory:")
        csv_files = pipeline.find_csv_files()
        
        datasets = set()
        tasks = set()
        formats = set()
        
        for csv_file in csv_files:
            parts = csv_file.relative_to(pipeline.converted_prompts_dir).parts
            if len(parts) >= 2:
                datasets.add(parts[0])
                tasks.add(parts[1])
            
            # Extract format from filename (e.g., answer_lookup_json_converted_prompts.csv)
            stem = csv_file.stem
            if '_' in stem:
                parts = stem.split('_')
                # Look for format in the parts
                for part in parts:
                    if part in ['json', 'xml', 'html', 'md', 'txt', 'ttl']:
                        formats.add(part)
                        break
        
        print(f"Datasets: {sorted(datasets)}")
        print(f"Tasks: {sorted(tasks)}")
        print(f"Formats: {sorted(formats)}")
        available_models = ["openai", "google"]  # Hardcoded since we didn't init clients
        print(f"Models: {available_models}")
        return 0
    
    # Run benchmark
    success = pipeline.run_benchmark(
        dataset=args.dataset,
        task=args.task,
        format_type=args.format,
        model=args.model,
        max_cases=args.max_cases,
        start_case=args.start_case,
        self_aug_type=args.self_aug
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETED")
    print("="*60)
    print(f"Status: {'SUCCESS' if success else 'FAILED'}")
    print(f"Models Used: {getattr(pipeline, 'models_used', list(pipeline.clients.keys()))}")
    if args.variants:
        model_dirs = [f"{model}_{args.variants}" for model in getattr(pipeline, 'models_used', list(pipeline.clients.keys()))]
        print(f"Results Saved To: benchmark_results/ (directories: {', '.join(model_dirs)})")
    else:
        print(f"Results Saved To: benchmark_results/")
    print("="*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())