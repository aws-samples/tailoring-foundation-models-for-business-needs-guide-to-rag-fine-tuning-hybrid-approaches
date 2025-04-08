from utils.helpers import json_to_jsonl, template_and_predict, logger 

import sagemaker

from sagemaker import Session
from sagemaker import Predictor
from sagemaker.s3 import S3Uploader
from sagemaker.s3 import S3Downloader
from sagemaker.jumpstart.estimator import JumpStartEstimator
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.session import Session
from sagemaker.exceptions import UnexpectedStatusException

from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

import os, json, boto3, time, shutil
from datetime import datetime, timezone, timedelta


class Finetuning():
    """
    A class to implement Finetuning in Sagemaker Jumpstart.
    
    This class handles the setup, execution, and testing of model finetuning using Amazon SageMaker JumpStart.
    It manages AWS role creation, data preparation, model training, and inference.
    
    Attributes:
        finetuning_method (str): The method used for finetuning
        model_id (str): The identifier of the base model to finetune
        model_name (str): Name to be given to the finetuned model
        bucket_name (str): S3 bucket name for storing training data and model artifacts
        template (dict): Template configuration for model input/output formatting
        num_epoch (int): Number of training epochs
        role_arn (str): ARN of the IAM role used for SageMaker execution
    """
    def __init__(self, bedrock_region: str, 
                finetuning_method: str, 
                model_id: str,
                model_name: str,
                bucket_name: str,
                template: dict,
                num_epoch: int,
                finetuning_instance:str
                ):
        self.bedrock_region = bedrock_region
        self.finetuning_method = finetuning_method
        self.model_id = model_id
        self.model_name = model_name
        self.bucket_name = bucket_name
        self.template = template
        self.num_epoch = num_epoch
        self.finetuning_instance = finetuning_instance
    

        
        # Initialize AWS clients
        self._initialize_aws_clients(bedrock_region)
        
        # Get or create SageMaker execution role
        self.role_arn = self._get_or_create_sagemaker_role()
        logger.info(f"Using SageMaker role: {self.role_arn}")

    def _initialize_aws_clients(self, region: str) -> None:
        """
        Initialize AWS clients with specified region.

        Args:
            region (str): AWS region name for client initialization
        """
        # Initialize boto3 clients
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.iam_client = boto3.client('iam', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        
        # Initialize SageMaker session
        self.sagemaker_session = sagemaker.Session(
            boto_session=boto3.Session(region_name=region),
            sagemaker_client=self.sagemaker_client
        )
    
    def _get_or_create_sagemaker_role(self) -> str:
        """
        Get existing SageMaker execution role or create a new one.
        
        Returns:
            str: ARN of the SageMaker execution role
        """
        ROLE_NAME = 'SageMakerExecutionRole'
        
        try:
            # Try to get existing role
            response = self.iam_client.get_role(RoleName=ROLE_NAME)
            print(f"Found existing SageMaker role: {ROLE_NAME}")
            return response['Role']['Arn']
            
        except self.iam_client.exceptions.NoSuchEntityException:
            print(f"Creating new SageMaker role: {ROLE_NAME}")
            return self._create_sagemaker_role(ROLE_NAME)
    
    def _create_sagemaker_role(self, role_name: str) -> str:
        """
        Create a new SageMaker execution role.
        
        Args:
            role_name: Name for the new role
            
        Returns:
            str: ARN of the created role
        """
        try:
            # Create trust policy
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "sagemaker.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            }
            
            # Create the role
            response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Execution role for SageMaker training and inference'
            )
            
            # Attach required policies
            required_policies = [
                'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
                'arn:aws:iam::aws:policy/AmazonS3FullAccess'  # Add S3 access for data
            ]
            
            for policy_arn in required_policies:
                try:
                    self.iam_client.attach_role_policy(
                        RoleName=role_name,
                        PolicyArn=policy_arn
                    )
                    print(f"Attached policy: {policy_arn}")
                except Exception as e:
                    print(f"Warning: Could not attach policy {policy_arn}: {str(e)}")
             
            return response['Role']['Arn']
            
        except Exception as e:
            print(f"Error creating SageMaker role: {str(e)}")
            raise


    def prepare_data_finetuning(self):
        """
        Prepare and upload training data for finetuning.

        Converts JSON data to JSONL format and uploads to S3 along with the template.
        Creates necessary directory structure and handles data transformation.

        Returns:
            str: S3 location of the prepared training data
        """
        os.makedirs(f'data/{self.finetuning_method}', exist_ok=True)
        data_location = f"s3://{self.bucket_name}/{self.finetuning_method}"
        if self.finetuning_method == "instruction_finetuning":
            local_data_file_train = f'data/{self.finetuning_method}/train.jsonl'
            json_to_jsonl(f'data/train/{self.finetuning_method}_train.json', local_data_file_train)
            with open(f"data/{self.finetuning_method}/template.json", "w") as f: #template is defined in config
                json.dump(self.template, f)
            S3Uploader.upload(f"data/{self.finetuning_method}/template.json", data_location)
        else: #training dataset for domain adaptation in txt format. 
            local_data_file_train = f"data/{self.finetuning_method}/train.txt"
            shutil.copyfile(f'data/train/{self.finetuning_method}_train.txt', local_data_file_train)
        
        local_data_file_test = f'data/{self.finetuning_method}/test.jsonl'
        json_to_jsonl(f'data/test/test.json', local_data_file_test) #same for instruction finetuning and domain adaptation

        S3Uploader.upload(local_data_file_train, data_location)
        S3Uploader.upload(local_data_file_test, data_location)
        return data_location

    def save_model_info(self, training_job_name: str, model_data_url: str) -> None:
        """
        Save model information for later deployment.
        
        Args:
            training_job_name (str): Name of the completed training job
            model_data_url (str): S3 URL where model artifacts are stored
        """
        model_info = {
            'training_job_name': training_job_name,
            'model_data_url': model_data_url,
            'model_id': self.model_id,
            'model_name': self.model_name,
            'creation_time': datetime.now().isoformat()
        }
        
        # Save to a JSON file
        os.makedirs('data/output/model_info', exist_ok=True)
        with open(f'data/output/model_info/{self.model_name}_info.json', 'w') as f:
            json.dump(model_info, f, indent=4)
        
        print(f"Model information saved to data/output/model_info/{self.model_name}_info.json")

    def create_endpoint_from_saved_model(self, model_name: str) -> Predictor:
        """
        Create an endpoint using previously saved model artifacts.
        
        Args:
            model_name (str): Name of the saved model to deploy
            
        Returns:
            sagemaker.predictor.Predictor: Predictor object for the deployed endpoint
            
        """
        # Load model information
        with open(f'data/output/model_info/{model_name}_info.json', 'r') as f:
            model_info = json.load(f)
            
        # Get the training job description to get model artifacts
        training_job = self.sagemaker_client.describe_training_job(
            TrainingJobName=model_info['training_job_name']
        )

        model = JumpStartModel(
            model_id=model_info['model_id'],
            model_version='2.2.2',
            region=self.bedrock_region,
            model_data={"S3DataSource": model_info['model_data_url']['S3DataSource']},
            role = self.role_arn
        )

        endpoint_name = f'{model_name}-endpoint'.replace('_', '-')

        # Deploy the model
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type="ml.g5.12xlarge", 
            endpoint_name=endpoint_name,
        )

        return predictor

    def finetune_model(self, train_data_location, deploy: bool = True):
        """
        Execute model finetuning using SageMaker JumpStart.
        
        Args:
            train_data_location (str): S3 location of the training data
            deploy (bool): Whether to deploy the model after training
            
        Returns:
            sagemaker.predictor.Predictor or None: Predictor object if deploy=True, None otherwise
        """
        output_path = f"s3://{self.bucket_name}/{self.finetuning_method}/output/jumpstart-{self.model_name}/"
        start_time_training = time.time()  
        if self.finetuning_method == "instruction_finetuning":
            instruction_label = "True"
        else:
            instruction_label = "False"

        estimator = JumpStartEstimator(
            model_id=self.model_id,
            model_version='2.2.2',
            environment={"accept_eula": "true"},
            disable_output_compression=True,
            hyperparameters={
                "instruction_tuned": instruction_label,
                "chat_dataset": "False",
                "epoch": self.num_epoch,
                "max_input_length": "4096",
            },
            instance_type=self.finetuning_instance,
            sagemaker_session=self.sagemaker_session,
            output_path=output_path,
            role=self.role_arn
        )

        # Train the model

        try:
            estimator.fit({"training": train_data_location})
        except UnexpectedStatusException as e:
            if "AlgorithmError: ExecuteUserScriptError" in str(e):
                logger.info("First attempt failed with ExecuteUserScriptError. Retrying once...")
                try:
                    estimator.fit({"training": train_data_location})
                except Exception as retry_error:
                    logger.error(f"Second attempt also failed: {str(retry_error)}")
                    raise
            else:
                logger.error(f"Training failed with unexpected status: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise
        
        # Save model information for later use
        self.save_model_info(
            training_job_name=estimator.latest_training_job.name,
            model_data_url=estimator.model_data
        )
        end_time_training = time.time()

        # Deploy only if requested
        if deploy:
            predictor = estimator.deploy(
                initial_instance_count=1,
                instance_type=self.finetuning_instance,
                container_startup_health_check_timeout=240
            )
            end_time_training_deploying = time.time()
            training_deployment_time = end_time_training_deploying - start_time_training
            return predictor, training_deployment_time

        training_time = end_time_training - start_time_training
        return None, training_time
    
    def delete_endpoint(self, predictor: Predictor = None, endpoint_name: str = None) -> None:
        """
        Delete the endpoint and associated resources.
        
        Args:
            predictor (sagemaker.predictor.Predictor, optional): Predictor object for the endpoint
            endpoint_name (str, optional): Name of the endpoint to delete
            
        Note:
            At least one of predictor or endpoint_name must be provided.
            If both are provided, predictor takes precedence.
            
        Raises:
            ValueError: If neither predictor nor endpoint_name is provided
        """
        try:
            if predictor is None and endpoint_name is None:
                raise ValueError("Either predictor or endpoint_name must be provided")

            if predictor:
                # Get endpoint name from predictor before deletion
                endpoint_name = predictor.endpoint_name
                # Delete endpoint and model using predictor
                predictor.delete_endpoint()
                predictor.delete_model()
            else:
                # Delete endpoint using SageMaker client
                self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                
                # Delete endpoint configuration
                try:
                    self.sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
                except self.sagemaker_client.exceptions.ClientError as e:
                    print(f"Warning: Could not delete endpoint configuration: {str(e)}")
                
                # Try to delete the model with the same name
                try:
                    self.sagemaker_client.delete_model(ModelName=endpoint_name)
                except self.sagemaker_client.exceptions.ClientError as e:
                    print(f"Warning: Could not delete model: {str(e)}")

            print(f"Endpoint '{endpoint_name}' and associated resources deleted successfully")
            
        except Exception as e:
            print(f"Error deleting endpoint: {str(e)}")
            raise

    def test_finetuned_model(self,predictor,endpoint_name):
        """
        Test the finetuned model with test dataset.

        Can work with either a predictor object or an endpoint name.
        Processes test data and saves results to a JSON file.

        Args:
            predictor (sagemaker.predictor.Predictor, optional): Predictor object for model endpoint
            endpoint_name (str, optional): Name of the deployed model endpoint

        Note:
            Either predictor or endpoint_name must be provided
        """

        counter = 1
        test_data_path = f"data/{self.finetuning_method}/test.jsonl"

        if predictor is None and endpoint_name is not None:
            # Create a Predictor instance
            predictor = Predictor(
                endpoint_name=endpoint_name,
                serializer=JSONSerializer(),  
                deserializer=JSONDeserializer(),
            )

           
        with open(test_data_path, 'r') as file:
            results = []
            inference_times = []
            for line in file:
                step_start_1 = time.time()
                # Parse each line as a JSON object
                product_data = json.loads(line)
                
                question = product_data.get("question")
                ground_truth = product_data.get("answer")
                start_time = time.time() 
                input_text, ground_truth, llm_response = template_and_predict(predictor, self.template, question,"", ground_truth)
                end_time = time.time()
                inference_time = end_time - start_time
                inference_times.append(inference_time)
                try:
                    llm_response  = llm_response['generated_text']
                except Exception as e:
                    logger.Error("Error! Llm response does not have generated_text field")

                results_dict = {
                    'input_text': input_text,
                    'ground_truth': ground_truth,
                    'llm_response': llm_response,
                }
                results.append(results_dict)
        avg_inference_time = sum(inference_times)/ len(inference_times)

        with open( f"data/output/{self.finetuning_method}_results.json", 'w') as json_file:
            json.dump(results, json_file, indent=4)
        
        
        return avg_inference_time
        
                
