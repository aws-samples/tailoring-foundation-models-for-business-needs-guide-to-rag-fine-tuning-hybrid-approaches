
from sagemaker import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

from utils.bedrock import BedrockHandler

import os, json, boto3, glob, time

from utils.helpers import json_to_jsonl, template_and_predict

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
                model_id,
                template
                ):
        """

        """
        self.predictor = predictor
        self.endpoint_name = endpoint_name
        self.rag_obj = rag_obj
        self.finetuning_obj = finetuning_obj
        self.knowledge_base_id = knowledge_base_id
        self.template = template

        self.bedrock_handler = BedrockHandler(
            self.rag_obj.bedrock_runtime, model_id
        )


    def test_hybrid_model(self):
        test_data_dir = f"data/test/"
        json_files = glob.glob(os.path.join(test_data_dir, "*.json"))
        if len(json_files) != 1:
            raise ValueError("There should be exactly one JSON file in the directory.")
        test_data = json_files[0]

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
            )

        with open(test_data, 'r') as file:
            data = json.load(file)
            results = []
            inference_times = []
            for product_data in data:
                # Parse each line as a JSON object
                question = product_data.get("question")
                ground_truth = product_data.get("answer")

                start_time = time.time()
                context = self.rag_obj.get_context(self.knowledge_base_id, question)
                input_text, ground_truth, llm_response = template_and_predict(self.predictor, self.template, question, context, ground_truth)
                end_time = time.time()
                inference_time = end_time - start_time
                inference_times.append(inference_time)
                try:
                    llm_response  = llm_response['generated_text']
                except Exception as e:
                    logger.error("Error! Llm responce does not have generated_text field")


                results_dict = {
                    'input_text': input_text,
                    'ground_truth': ground_truth,
                    'llm_response': llm_response,
                    'context': context
                }
                results.append(results_dict)

        avg_inference_time = sum(inference_times)/ len(inference_times)
                    
        with open( f"data/output/hybrid_results.json", 'w') as json_file:
            json.dump(results, json_file, indent=4)
        return avg_inference_time
        
