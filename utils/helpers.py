import logging, boto3, os, json, re
from typing import Dict, List, Tuple
import pandas as pd


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Get a shared logger instance
logger = logging.getLogger("app_logger")


def upload_data_S3(s3_client, data_folder_path, kb_data_folder, bucket_name):
    for file_name in os.listdir(kb_data_folder):
        file_path = os.path.join(kb_data_folder, file_name)
        if os.path.isfile(file_path):  # Check if it's a file (skip directories)
            try:
                # Upload the file to S3
                s3_key = f"kb-data/{file_name}"
                s3_client.upload_file(file_path, bucket_name, s3_key)
                print(f"Uploaded {file_name} to s3://{bucket_name}/kb-data/{file_name}")
            except Exception as e:
                print(f"Failed to upload {file_name}: {str(e)}")
        else:
            print("WARNING! There is nothing to upload!!")

def json_to_jsonl(json_file_path, output_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    with open(output_file_path, 'w') as outfile:
        for record in data:
            json.dump(record, outfile)
            outfile.write('\n') 

def template_and_predict(predictor, template, question, context, ground_truth, input_output_demarkation_key="\n\n### Response:\n"):

    inputs = template["prompt"].format(question=question, context=context)
    inputs += input_output_demarkation_key
    payload = {"inputs": inputs, "parameters": {"max_new_tokens": 4096}}

    response = predictor.predict(payload)
    return inputs, ground_truth, response

def load_json_file(file_path: str) -> List[Dict]:
        """Load and parse a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

def get_stack_outputs(stack_name: str, region: str) -> dict:
    """
    Get CloudFormation stack outputs
    
    Args:
        stack_name: Name of the CloudFormation stack
        region: AWS region
    
    Returns:
        dict: Dictionary of stack outputs
    """
    cloudformation = boto3.client('cloudformation', region_name=region)
    response = cloudformation.describe_stacks(StackName=stack_name)
    outputs = response['Stacks'][0]['Outputs']
    
    return {output['OutputKey']: output['OutputValue'] for output in outputs}

def create_summary_table(output_dir="data/output", summary_file="summary_results.csv"):
    """
    Creates a summary table with average scores from the three JSON files.
    
    Args:
        output_dir (str): Directory containing the JSON files
        summary_file (str): Name of the output summary file
    """
    # Dictionary to store results
    results = {
        'method': [],
        'avg_bert_score': [],
        'avg_llm_evaluator_score': []
    }
    
    # List of files to process
    files = ['rag_results.json', 'instruction_finetuning_results.json', 'hybrid_results.json']
    
    # Process each file
    for file in files:
        file_path = os.path.join(output_dir, file)
        method = file.replace('_results.json', '') # Extract method name from filename
                
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Calculate averages
            bert_scores = [sample.get('bert_score', 0) for sample in data]
            llm_scores = [sample.get('llm_evaluator_score', 0) for sample in data]
            
            avg_bert = sum(bert_scores) / len(bert_scores) if bert_scores else 0
            avg_llm = sum(llm_scores) / len(llm_scores) if llm_scores else 0
            
            # Store results
            results['method'].append(method)
            results['avg_bert_score'].append(round(avg_bert, 4))
            results['avg_llm_evaluator_score'].append(round(avg_llm, 4))
            
        except FileNotFoundError:
            print(f"Warning: {file} not found in {output_dir}")
        except json.JSONDecodeError:
            print(f"Warning: Error decoding {file}")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, summary_file)
    df.to_csv(output_path, index=False)
    
    print(f"\nSummary table created at: {output_path}")
    print("\nResults summary:")
    print(df.to_string())
    
    return df