import base64
import json
from typing import Optional
import os, boto3
from utils.bedrock import BedrockHandler, KBHandler
from src.evaluation import evaluate

class Rag:
    """
    A class to implement RAG with Knowledge Bases.
    """
    def __init__(self, bedrock_region: str, kb_configs: dict):
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

        self.kb_configs = kb_configs



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
        test_data_path = f"data/test/qa_dataset_test.json" # TODO make the json file name generic

        with open(test_data_path, 'r') as file:
            data = json.load(file)
            results = []
            for product_data in data:                
                print(f"test data no: {counter}")
                # Parse each line as a JSON object
                question = product_data.get("question")
                ground_truth = product_data.get("answer")

                bedrock_messages = []
                context = self.get_context(knowledge_base_id, question)

                bedrock_messages.append(
                    bedrock_handler.user_message(question, context)
                )
                response = bedrock_handler.invoke_model(bedrock_messages)
                response_text = response['output']['message']['content'][0]['text']

                results_dict = {
                    'input_text': question,
                    'ground_truth': ground_truth,
                    'llm_response': response_text,
                    'context': context
                }
                results.append(results_dict)

                print(f'Input: {question}')
                print(f'Ground_truth: {ground_truth}')
                print(f'LLM response: {response_text}')
                
                print(f'Context: {context}')
                counter += 1

        results_file_path = "data/output/rag_results.json"
        os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

        with open(results_file_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)
                    