from utils.helpers import json_to_jsonl, template_and_predict, update_trust_relationship

from sagemaker import Session
from sagemaker import Predictor
from sagemaker.s3 import S3Uploader
from sagemaker.s3 import S3Downloader
from sagemaker.jumpstart.estimator import JumpStartEstimator

from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer


import os, json, boto3

class Finetuning():
    """
    A class to implement Finetuning in Sagemaker Jumpstart
    """
    def __init__(self, bedrock_region: str, 
                finetuning_method: str, 
                model_id: str,
                bucket_name:str,
                ):
        """

        """
        self.finetuning_method = finetuning_method
        self.local_template_file = f"data/{finetuning_method}/template.json"
        self.model_id = model_id

        self.iam_client = boto3.client(
        service_name="iam", 
        region_name=bedrock_region 
        )

        self.sagemaker_session = Session()
        self.bucket_name = bucket_name 
        """
        update_trust_relationship(
            iam_client=self.iam_client, 
            role_name="Admin", 
            principal_service="lambda.amazonaws.com",
            actions=["sts:AssumeRole"]
        )
        """


    def prepare_data_finetuning(self):
        os.makedirs(f'data/{self.finetuning_method}', exist_ok=True)

        json_to_jsonl('data/train/qa_dataset_train.json', f'data/{self.finetuning_method}/train.jsonl')
        json_to_jsonl('data/test/qa_dataset_test.json', f'data/{self.finetuning_method}/test.jsonl')

        # TODO: place this into config
        template = {
            "prompt": (
                "You are a helpful AI assistant. You are intelligent and concise."
                " Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                " If you don't know the answer, just say, \"I don't know.\" Don't make anything up."
                " ### Instruction:\n Answer the customer's question: {question}\n\n"
            ),
            "completion": "{answer}",
        }

        with open(self.local_template_file, "w") as f:
            json.dump(template, f)

        data_location = f"s3://{self.bucket_name}/{self.finetuning_method}"
        local_data_file_train = f"data/{self.finetuning_method}/train.jsonl"
        local_data_file_test = f"data/{self.finetuning_method}/test.jsonl"

        S3Uploader.upload(local_data_file_train, data_location)
        S3Uploader.upload(local_data_file_test, data_location)
        S3Uploader.upload(f"data/{self.finetuning_method}/template.json", data_location)
        return data_location
    
    def finetune_model(self,train_data_location):
        # TODO: Get hyperparameters from config
        output_path = f"s3://{self.bucket_name}/{self.finetuning_method}/output/jumpstart-llama3-instruction/"

        estimator = JumpStartEstimator(
            model_id=self.model_id,
            model_version = '2.2.2',
            environment={"accept_eula": "true"},  # set "accept_eula": "true" to accept the EULA for gated models
            disable_output_compression=True,
            hyperparameters={
                "instruction_tuned": "True",
                "chat_dataset": "False",
                "epoch": "3",
                "max_input_length": "4096",
            },
            instance_type= 'ml.g5.12xlarge', # default value
            sagemaker_session=self.sagemaker_session,
            output_path=output_path,
            role="arn:aws:iam::339712995635:role/Admin" # TODO: role should not be hardcoded
        )

        estimator.fit({"training": train_data_location})

        predictor = estimator.deploy(
                initial_instance_count=1, 
                instance_type='ml.g4dn.12xlarge', 
                container_startup_health_check_timeout=900
            )
        # 'ml.g5.8xlarge' worked for finetuning results, not for hybrid
        # 'ml.p3.8xlarge' and 'ml.g4dn.12xlarge' gave an error

        return predictor

    # It can work with predictor directly or with endpoint_name
    def test_finetuned_model(self,predictor,endpoint_name):
        counter = 1
        test_data_path = f"data/{self.finetuning_method}/test.jsonl"

        if predictor is None and endpoint_name is not None:
            # Create a Predictor instance
            predictor = Predictor(
                endpoint_name=endpoint_name,
                serializer=JSONSerializer(),  
                deserializer=JSONDeserializer(),
                #component_name = 'variant-1'
            )
            
        with open(self.local_template_file, "r") as file:
            template = json.load(file)

        with open(test_data_path, 'r') as file:
            results = []
            for line in file:
                print(f"test data no: {counter}")
                # Parse each line as a JSON object
                product_data = json.loads(line)
                
                question = product_data.get("question")
                ground_truth = product_data.get("answer")

                input_text, ground_truth, llm_response = template_and_predict(predictor, template, question,"", ground_truth)
                
                try:
                    llm_response  = llm_response['generated_text']
                except Exception as e:
                    print("WARNING! Llm responce does not have generated_text field")

                results_dict = {
                    'input_text': input_text,
                    'ground_truth': ground_truth,
                    'llm_response': llm_response,
                }
                results.append(results_dict)

                print(f'Input: {input_text}')
                print(f'Ground_truth: {ground_truth}')
                print(f'LLM response: {llm_response}')
                counter += 1

        with open( f"data/output/{self.finetuning_method}_results.json", 'w') as json_file:
            json.dump(results, json_file, indent=4)
        
                
