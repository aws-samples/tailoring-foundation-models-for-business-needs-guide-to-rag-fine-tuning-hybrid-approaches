import base64
import json
from typing import Optional
import os, boto3, time, glob
from utils.bedrock import BedrockHandler, KBHandler
from datetime import datetime, timezone, timedelta

class Rag:
    """
    A class to implement RAG with Knowledge Bases.
    """
    def __init__(self, bedrock_region: str, kb_configs: dict, rag_template: dict):
        """
        Initialize the RAG class with required configurations.
        
        Args:
            bedrock_region (str): AWS region for Bedrock.
            kb_configs (dict): Knowledge base configuration parameters.
        """

        self.bedrock_agent_runtime_client = boto3.client(
            service_name="bedrock-agent-runtime", region_name=bedrock_region
        )

        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime", region_name=bedrock_region
        )

        self.bedrock_agent = boto3.client(
            'bedrock-agent', region_name=bedrock_region
        )

        self.kb_configs = kb_configs
        self.rag_template = rag_template["prompt"]
    
    def wait_for_kb_sync(self, knowledge_base_id: str, data_source_id: str, max_wait_time: int = 900) -> bool:
        """
        Wait for knowledge base sync to complete.
        """
        timezone_offset = timezone(timedelta(hours=1))
        
        try:
            # Get the most recent job
            response = self.bedrock_agent.list_ingestion_jobs(  # Note the _runtime suffix
                knowledgeBaseId=knowledge_base_id,
                dataSourceId=data_source_id,
                maxResults=1
            )
            
            jobs = response.get('ingestionJobSummaries', [])  # Note: different key name
            if not jobs:
                print("No ingestion jobs found")
                return True
                
            latest_job = jobs[0]
            
            # Check if job recently succeeded
            if latest_job['status'] == 'SUCCEEDED':
                job_end_time = latest_job.get('completionTime', datetime.now(timezone_offset))  # Note: different key name
                if isinstance(job_end_time, str):
                    job_end_time = datetime.fromisoformat(job_end_time.replace('Z', '+00:00')).astimezone(timezone_offset)
                
                time_since_completion = (datetime.now(timezone_offset) - job_end_time).total_seconds()
                if time_since_completion < 2:
                    print(f"Recent successful sync found (completed {time_since_completion:.0f} seconds ago)")
                    return True
            
            # Wait for in-progress job
            if latest_job['status'] == 'IN_PROGRESS':
                print(f"Found active ingestion job: {latest_job['ingestionJobId']}")  # Note: different key name
                start_time = time.time()
                
                while (time.time() - start_time) < max_wait_time:
                    job_response = self.bedrock_agent.get_ingestion_job(  # Note the _runtime suffix
                        knowledgeBaseId=knowledge_base_id,
                        dataSourceId=data_source_id,
                        ingestionJobId=latest_job['ingestionJobId']  # Note: different key name
                    )
                    
                    status = job_response['status']
                    print(f"Ingestion job status: {status}")
                    
                    if status == 'SUCCEEDED':
                        print("Knowledge base sync completed successfully")
                        return True
                    elif status in ['FAILED', 'CANCELLED']:
                        print(f"Ingestion job failed with status: {status}")
                        print(f"Error message: {job_response.get('failureReason', 'No error message provided')}")  # Note: different key name
                        return False
                        
                    time.sleep(30)
                
                print(f"Timeout waiting for knowledge base sync after {max_wait_time} seconds")
                return False
            
            return True
                
        except Exception as e:
            print(f"Error checking sync status: {str(e)}")
            return False

    def get_context(self, kb_id: str, prompt: str) -> str:
        """
        Retrieves the relevant context from the knowledge base based on the prompt.

        Args:
            kb_id (str): Knowledge base ID.
            prompt (str): User prompt for retrieval.

        Returns:
            str: Retrieved context as a string.
        """
        # Initialize retriever (KB Handler)
        retriever = KBHandler(
            self.bedrock_agent_runtime_client, self.kb_configs, kb_id=kb_id
        )
        
        # Retrieve documents from the knowledge base
        docs = retriever.get_relevant_docs(prompt)
        
        # Parse the knowledge base output to a string
        context = retriever.parse_kb_output_to_string(docs)
        
        return context

    def test_rag(self,knowledge_base_id, model_name, model_id):
        bedrock_handler = BedrockHandler(
            self.bedrock_runtime, model_id
        )

        counter = 1

        test_data_dir = f"data/test/"
        json_files = glob.glob(os.path.join(test_data_dir, "*.json"))
        if len(json_files) != 1:
            raise ValueError("There should be exactly one JSON file in the directory.")
        test_data_path = json_files[0]

        with open(test_data_path, 'r') as file:
            data = json.load(file)
            results = []
            inference_times = []
            for product_data in data: 
                # Parse each line as a JSON object
                question = product_data.get("question")
                ground_truth = product_data.get("answer")

                bedrock_messages = []
                start_time = time.time()
                context = self.get_context(knowledge_base_id, question)

                prompt = self.rag_template.format(question=question, context=context)
                #print(f"======Prompt: {prompt}")

                bedrock_messages.append(
                    bedrock_handler.user_message(prompt)
                )

                response = bedrock_handler.invoke_model(bedrock_messages)
                end_time = time.time()
                inference_time = end_time - start_time
                inference_times.append(inference_time)
                #print(f"RESPONSE: {response}")
                response_text = response['output']['message']['content'][0]['text']

                results_dict = {
                    'input_text': question,
                    'ground_truth': ground_truth,
                    'llm_response': response_text,
                    'context': context
                }
                results.append(results_dict)
                """
                print(f'Input: {question}')
                print(f'Ground_truth: {ground_truth}')
                print(f'LLM response: {response_text}')
                
                print(f'Context: {context}')
                """

                counter += 1
        
        avg_inference_time = sum(inference_times)/ len(inference_times)

        results_file_path = "data/output/rag_results.json"
        os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

        with open(results_file_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)
            
        return avg_inference_time