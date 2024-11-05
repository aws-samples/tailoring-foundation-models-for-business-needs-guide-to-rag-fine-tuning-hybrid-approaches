#!/usr/bin/env python3
import os,sys,subprocess,time

import aws_cdk as cdk
from utils.build_infra import build_kb, delete_all
from constructs import DependencyGroup

from config import EnvSettings, KbConfig, DsConfig, RAGConfig, FinetuningConfig

from infrastructure.stacks.kb_role_stack import KbRoleStack
from infrastructure.stacks.oss_infra_stack import OpenSearchServerlessInfraStack
from infrastructure.stacks.kb_infra_stack import KbInfraStack
from infrastructure.stacks.s3_stack import S3Stack
from utils.helpers import logger, upload_data_S3
from src import rag
from src import finetuning
from src import hybrid
import boto3
from sagemaker.s3 import S3Uploader

from utils.helpers import json_to_jsonl, template_and_predict

region = EnvSettings.ACCOUNT_REGION
account_id = EnvSettings.ACCOUNT_ID

kb_role_name = KbConfig.KB_ROLE_NAME
kb_name = KbConfig.KB_NAME

bucket_name = DsConfig.S3_BUCKET_NAME
kb_data_folder = DsConfig.KB_DATA_FOLDER

number_of_results = RAGConfig.NUMBER_OF_RESULTS
model_id_finetuning = FinetuningConfig.MODEL_ID
model_name_finetuning = FinetuningConfig.MODEL_NAME
finetuning_method = FinetuningConfig.METHOD

model_id_rag = RAGConfig.MODEL_ID
model_name_rag = RAGConfig.MODEL_NAME

s3_client = boto3.client('s3', region_name=region) 
data_folder_path = "data"


if __name__ == "__main__":
    logger.info("Starting the application...")
    logger.info("START - Build Knowledge Base")

    
    #knowledge_base_id = build_kb()
    knowledge_base_id = 'YBQBDFRQSJ' #TODO: If you want to skip the kb creation, set the correct kb_id and commented out above line

    logger.info("FINISH - Build Knowledge Base")

    #upload the data to s3    
    logger.info("START - KB Data Load into S3")
    kb_data_folder = f'{data_folder_path}/kb-data'
    upload_data_S3(s3_client, data_folder_path, kb_data_folder, bucket_name)
    logger.info("FINISH - KB Data Load into S3")

    logger.info("START - Waiting for data sync for KB")
    time.sleep(60) # Waiting for 1 min, to make sure that the data is sync, TODO: Find a better way
    logger.info("FINISH - Waiting for data sync for KB")

    kb_configs = {
        "vectorSearchConfiguration": {
            "numberOfResults": number_of_results 
        }
    }

    rag_obj = rag.Rag(
        bedrock_region=region,
        kb_configs=kb_configs,
    )

    #rag_obj.test_rag(knowledge_base_id,model_name_rag, model_id_rag)
    

    
    finetuning_obj = finetuning.Finetuning(
        bedrock_region=region,
        finetuning_method = finetuning_method,
        model_id = model_id_finetuning,
        bucket_name = bucket_name

    )

    data_location = finetuning_obj.prepare_data_finetuning()
    logger.info("FINISH - Prepare_data_finetuning")

    predictor = finetuning_obj.finetune_model(data_location)
    logger.info("FINISH - Finetune Model")

    #endpoint_name = "llama-3-1-8b-instruct-2024-10-31-15-14-54-211"  #TODO: If you want to use already deployed model, find the correct endpoint name and commented out above 4 lines and  
    finetuning_obj.test_finetuned_model(predictor,None)

    #finetuning_obj.test_finetuned_model(None, endpoint_name)

    logger.info("FINISH - Test Finetune model")


    hybrid_obj = hybrid.Hybrid(
        predictor, # predictor
        None, #endpoint_name,
        rag_obj,
        finetuning_obj,
        knowledge_base_id,
        model_id_rag
    )

    hybrid_obj.test_hybrid_model()
    

    #create clean-up scripts 

    #delete_all()