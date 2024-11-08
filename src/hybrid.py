
from sagemaker import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

from utils.bedrock import BedrockHandler

import os, json, boto3
from utils.helpers import json_to_jsonl, template_and_predict
from src.evaluation import evaluate

from botocore.config import Config
from sagemaker import Session




class Hybrid():
    """
    A class to implement Hybrid approach, uses finetuned model with RAG approach
    """
    def __init__(self, 
                predictor,
                endpoint_name,
                rag_obj,
                finetuning_obj,
                knowledge_base_id,
                model_id
                ):
        """

        """
        self.predictor = predictor
        self.endpoint_name = endpoint_name
        self.rag_obj = rag_obj
        self.finetuning_obj = finetuning_obj
        self.knowledge_base_id = knowledge_base_id

        self.bedrock_handler = BedrockHandler(
            self.rag_obj.bedrock_runtime, model_id
        )




    # It can work with predictor directly or with endpoint_name
    def test_hybrid_model(self):
        counter = 1
        test_data_path = f"data/test/qa_dataset_test.json" # TODO make the json file name generic
        config = Config(
            connect_timeout=300,  # Time in seconds to establish the connection
            read_timeout=300     # Time in seconds to wait for a response
        )

       # Create a SageMaker runtime client with the custom configuration
        sagemaker_runtime_client = boto3.client("sagemaker-runtime", config=config)

        # Initialize the SageMaker session with the customized runtime client
        sagemaker_session = Session(sagemaker_runtime_client=sagemaker_runtime_client)

        if self.predictor is None and self.endpoint_name is not None:
            # Create a Predictor instance
            print("Predictor is created using endpoint")
            self.predictor = Predictor(
                endpoint_name=self.endpoint_name,
                sagemaker_session=sagemaker_session,
                serializer=JSONSerializer(),  
                deserializer=JSONDeserializer(),
                #component_name = 'variant-1'
            )
       
        """
        Predictor: {'endpoint_name': 'llama-3-1-8b-instruct-2024-10-29-16-19-08-933', 'sagemaker_session': <sagemaker.session.Session object at 0x3368af580>, 'serializer': <sagemaker.base_serializers.JSONSerializer object at 0x335b1ad60>, 'deserializer': <sagemaker.base_deserializers.JSONDeserializer object at 0x3368af490>}
        """
        
        # TODO define template in config
        template = {
            "prompt": (
                "You are a helpful AI assistant. You are intelligent and concise. Below is an instruction that describes a task, "
                "paired with an input that provides further context.\n"
                "Write a response that appropriately completes the request.\n\n"
                "If you don't know the answer, just say, \"I don't know.\" Don't make anything up.\n"
                "### Instruction:\n"
                "Answer the customer's question: {question}\n\n"
                "### Input:\n{context}\n\n"
            ),
            "completion": "{answer}"
        }         

        with open(test_data_path, 'r') as file:
            data = json.load(file)
            results = []
            for product_data in data:
                print(f"test data no: {counter}")
                # Parse each line as a JSON object
                question = product_data.get("question")
                ground_truth = product_data.get("answer")

                
                context = self.rag_obj.get_context(self.knowledge_base_id, question)

                input_text, ground_truth, llm_response = template_and_predict(self.predictor, template, question, context, ground_truth)
                try:
                    llm_response  = llm_response['generated_text']
                except Exception as e:
                    print("WARNING! Llm responce does not have generated_text field")

                """
                bedrock_messages = []
                bedrock_messages.append(
                    self.bedrock_handler.user_message(question, context)
                )
                payload = bedrock_messages[0]['content'][0]["text"]
                payload += "### Response:"
                """

                """ PAYLOAD format:
                {'inputs': 'You are a helpful AI assistant. You are intelligent and concise. Below is an instruction that describes a task. 
                Write a response that appropriately completes the request.\n\n If you don\'t know the answer, just say, "I don\'t know." Don\'t 
                make anything up. ### Instruction:\n Answer the customer\'s question: What are the key safety considerations and precautions 
                outlined in the safety instructions for the MANUFLEX 9000?\n\n\n\n### Response:\n', 'parameters': {'max_new_tokens': 4096}}
                """

                results_dict = {
                    'input_text': input_text,
                    'ground_truth': ground_truth,
                    'llm_response': llm_response,
                    'context': context
                }
                results.append(results_dict)

                print(f'Input: {input_text}')
                print(f'Ground_truth: {ground_truth}')
                print(f'LLM response: {llm_response}')
                counter += 1
                
        # TODO: create a results folder and put all the results there
        with open( f"data/output/hybrid_results.json", 'w') as json_file:
            json.dump(results, json_file, indent=4)
        
