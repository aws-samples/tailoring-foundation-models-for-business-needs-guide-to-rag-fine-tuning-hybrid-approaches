#!/usr/bin/env python3
import os,sys,subprocess,time

import aws_cdk as cdk
from utils.build_infra import build_kb, delete_all
from constructs import DependencyGroup

from config import EnvSettings, KbConfig, DsConfig, RAGConfig, FinetuningConfig, EvaluationConfig, Templates

from infrastructure.stacks.kb_role_stack import KbRoleStack
from infrastructure.stacks.oss_infra_stack import OpenSearchServerlessInfraStack
from infrastructure.stacks.kb_infra_stack import KbInfraStack
from infrastructure.stacks.s3_stack import S3Stack
from utils.helpers import logger, upload_data_S3, create_summary_table
from src import rag, finetuning, hybrid, llm_evaluator, evaluation

import boto3
from sagemaker.s3 import S3Uploader

from utils.helpers import json_to_jsonl, template_and_predict, get_stack_outputs



region = EnvSettings.ACCOUNT_REGION
account_id = EnvSettings.ACCOUNT_ID

kb_role_name = KbConfig.KB_ROLE_NAME
kb_name = KbConfig.KB_NAME

bucket_name = DsConfig.S3_BUCKET_NAME
kb_data_folder = DsConfig.KB_DATA_FOLDER

model_id_finetuning = FinetuningConfig.MODEL_ID
model_name_finetuning = FinetuningConfig.MODEL_NAME
finetuning_method = FinetuningConfig.METHOD
num_epoch = FinetuningConfig.NUM_EPOCH

number_of_results = RAGConfig.NUMBER_OF_RESULTS
model_id_rag = RAGConfig.MODEL_ID
model_name_rag = RAGConfig.MODEL_NAME

evaluator_models = EvaluationConfig.MODELS_EVAL
evaluator_prompt_template = EvaluationConfig.PROMPT_TEMPLATE
evaluator_score_pattern = EvaluationConfig.SCORE_PATTERN


finetuning_template = Templates.FINETUNING_TEMPLATE
hybrid_template = Templates.HYBRID_TEMPLATE
rag_template = Templates.RAG_TEMPLATE

s3_client = boto3.client('s3', region_name=region) 
data_folder_path = "data"


if __name__ == "__main__":
    logger.info("Starting the application...")
    stack_outputs = get_stack_outputs("KbInfraStack", region)

    knowledge_base_id = stack_outputs['KnowledgeBaseId']
    data_source_id = stack_outputs['DataSourceId']
    logger.info(f"Knowledge Base ID: {knowledge_base_id}")
    logger.info(f"Data Source ID: {data_source_id}")

    kb_configs = {
        "vectorSearchConfiguration": {
            "numberOfResults": number_of_results 
        }
    }

    rag_obj = rag.Rag(
        bedrock_region=region,
        kb_configs=kb_configs,
        rag_template = rag_template
    )
    kb_data_path = f'{data_folder_path}/{kb_data_folder}'
    upload_data_S3(s3_client, data_folder_path, kb_data_path, bucket_name)
    logger.info("START - Knowledge base sync")
    if not rag_obj.wait_for_kb_sync(
        knowledge_base_id=knowledge_base_id,
        data_source_id=data_source_id
    ):
        raise Exception("Knowledge base sync failed or timed out")
    logger.info("FINISH - Knowledge base sync")

    logger.info("START - Testing RAG")
    inference_time_rag = rag_obj.test_rag(knowledge_base_id,model_name_rag, model_id_rag)
    logger.info("FINISH - Testing RAG")
    
    finetuning_obj = finetuning.Finetuning(
        bedrock_region=region,
        finetuning_method = finetuning_method,
        model_id = model_id_finetuning,
        model_name = model_name_finetuning,
        bucket_name = bucket_name,
        template = finetuning_template,
        num_epoch = num_epoch

    )
    
    logger.info("START - Prepare_data_finetuning")
    data_location = finetuning_obj.prepare_data_finetuning()
    logger.info(f"INFO - Data location: {data_location}")
    logger.info("FINISH - Prepare_data_finetuning")

    logger.info("START - Finetune Model")
    predictor, training_time = finetuning_obj.finetune_model(data_location, True) # It will also deploy the model, if you want to deploy it later, change True to False
    logger.info(f'INFO - Trainig_time: {training_time:.2f} seconds')
    #predictor= finetuning_obj.create_endpoint_from_saved_model(model_name = "llama3_8b_instruct") # Use this line if you already finetuned the model but don't have the endpoint, instead of above line.
    logger.info("FINISH - Finetune Model")
    

    #endpoint_name = "llama-3-1-8b-instruct-2025-01-23-10-03-57-788" #TODO: If you want to use already deployed model, find the correct endpoint name
    logger.info("START - Testing FINETUNING")
    inference_time_finetuning = finetuning_obj.test_finetuned_model(predictor, None)
    #inference_time_finetuning = finetuning_obj.test_finetuned_model(None, endpoint_name) #TODO: If you want to use endpoint_name instead of predictor obj.

    logger.info("FINISH - Testing FINETUNING")

    hybrid_obj = hybrid.Hybrid(
        predictor, # predictor
        None, #endpoint_name eg. 'llama3-8b-instruct-endpoint',
        rag_obj,
        finetuning_obj,
        knowledge_base_id,
        model_id_rag,
        hybrid_template
    )

    logger.info("START - Testing RAG on Finetuned model")
    inference_time_hybrid = hybrid_obj.test_hybrid_model()
    logger.info("FINISH - Testing RAG on Finetuned model")


    eval_obj = evaluation.Evaluation(
        bedrock_region=region,
        evaluator_models = evaluator_models,
        evaluator_prompt_template = evaluator_prompt_template,
        score_pattern = evaluator_score_pattern
    )

    finetuning_results = f'data/output/{finetuning_method}_results.json'
    rag_results = 'data/output/rag_results.json'
    hybrid_results = 'data/output/hybrid_results.json'

    logger.info("START - Evaluation")

    scores = eval_obj.calculate_scores(
        finetuning_results,
        rag_results,
        hybrid_results
    )
    eval_obj.save_scores(
        scores,
        finetuning_results,
        rag_results,
        hybrid_results
    )

    eval_obj.calculate_aggregated_scores(
        finetuning_results,
        rag_results,
        hybrid_results
    )
    logger.info("FINISH - Evaluation")

    logger.info("START - Summary Table Creation")
    inference_times = {
        'rag': inference_time_rag,
        f'{finetuning_method}': inference_time_finetuning,
        'hybrid': inference_time_hybrid
    }
    print(inference_times)
    create_summary_table(inference_times, finetuning_method, "data/output","summary_results.csv")
    logger.info("FINISH - Summary Table Creation")
    
    
    #Clean-up
    #finetuning_obj.delete_endpoint(endpoint_name = 'llama3-8b-instruct-endpoint') #TODO: Replace with your actual endpoint name 
    #infra_dir = "./infrastructure"
    #run_command(f"cdk destroy --all", cwd=infra_dir) #TODO: Should we do this in code or should users do it from terminal
