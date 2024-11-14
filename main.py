#!/usr/bin/env python3
import os,sys,subprocess,time

import aws_cdk as cdk
from utils.build_infra import build_kb, delete_all
from constructs import DependencyGroup

from config import EnvSettings, KbConfig, DsConfig, RAGConfig, FinetuningConfig, EvaluationConfig

from infrastructure.stacks.kb_role_stack import KbRoleStack
from infrastructure.stacks.oss_infra_stack import OpenSearchServerlessInfraStack
from infrastructure.stacks.kb_infra_stack import KbInfraStack
from infrastructure.stacks.s3_stack import S3Stack
from utils.helpers import logger, upload_data_S3
from src import rag, finetuning, hybrid, llm_evaluator, evaluation

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

evaluator_models = EvaluationConfig.MODELS_EVAL

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
    #upload_data_S3(s3_client, data_folder_path, kb_data_folder, bucket_name)
    # logger.info("FINISH - KB Data Load into S3")

    # logger.info("START - Waiting for data sync for KB")
    # time.sleep(60) # Waiting for 1 min, to make sure that the data is sync, TODO: Find a better way
    # logger.info("FINISH - Waiting for data sync for KB")

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

    # data_location = finetuning_obj.prepare_data_finetuning()
    # logger.info("FINISH - Prepare_data_finetuning")

    # predictor = finetuning_obj.finetune_model(data_location)
    # logger.info("FINISH - Finetune Model")

    endpoint_name = "llama-3-1-8b-instruct-2024-10-31-15-14-54-211"  #TODO: If you want to use already deployed model, find the correct endpoint name and commented out above 4 lines and  
    #finetuning_obj.test_finetuned_model(predictor,None)

    #finetuning_obj.test_finetuned_model(None, endpoint_name)

    logger.info("FINISH - Test Finetune model")


    hybrid_obj = hybrid.Hybrid(
        None, # predictor
        endpoint_name, #endpoint_name,
        rag_obj,
        finetuning_obj,
        knowledge_base_id,
        model_id_rag
    )

    #hybrid_obj.test_hybrid_model()
    eval_obj = evaluation.Evaluation(
        bedrock_region=region,
        evaluator_models = evaluator_models
    )

    finetuning_results = 'data/output/instruction_finetuning_results.json'
    rag_results = 'data/output/rag_results.json'
    
    bert_score_finetuning, bert_score_rag, llm_evaluator_scores_finetuning, llm_evaluator_scores_rag = eval_obj.calculate_scores(finetuning_results, rag_results)
    print(bert_score_finetuning)
    print(bert_score_rag)
    print(llm_evaluator_scores_finetuning)
    print(llm_evaluator_scores_rag)

    eval_obj.save_scores(finetuning_results,rag_results, bert_score_finetuning, bert_score_rag, llm_evaluator_scores_finetuning, llm_evaluator_scores_rag)
    

    """
    eval_obj = llm_evaluator.LLMEvaluator(
        bedrock_region=region
    )
    ground_truth = "The MAPROTECH X5000 consists of three key components:\n\n1. Sensor Network: A network of high-precision sensors strategically placed throughout the production facilities to collect real-time data on various parameters like temperature, pressure, vibration, and energy consumption.\n\n2. Central Processing Unit (CPU): The CPU receives data from the sensor network, processes it using advanced analytics and machine learning algorithms to identify patterns, detect anomalies, and predict potential issues.\n\n3. User Interface: The system's intuitive user interface provides real-time data visualizations, customizable reports, and actionable insights to operators, engineers, and managers.\n\nThese components work together seamlessly. The sensor network collects data, which is transmitted wirelessly to the CPU for processing. The CPU analyzes the data and generates insights, which are then presented through the user interface, allowing users to monitor and control the production processes effectively"
    text1 = "The MAPROTECH X5000 Industrial Process Monitoring System is a comprehensive monitoring solution designed for industrial processes. The key components of this system include:\n\n1.  **Data Acquisition Units (DAUs):** These units are responsible for collecting data from various sensors and transmitters in the industrial process. They convert the raw data into a digital format that can be processed by the system.\n2.  **Communication Network:** The DAUs transmit the collected data to a central server or a network of servers through a communication network. This network can be a wired or wireless network, depending on the specific requirements of the industrial process.\n3.  **Server and Software:** The central server or network of servers processes the data received from the DAUs. The server runs the MAPROTECH X5000 software, which provides a user-friendly interface for monitoring and analyzing the industrial process.\n4.  **Human-Machine Interface (HMI):** The HMI is the user interface of the system, which provides real-time data and trends of the industrial process. It allows operators to monitor the process, set alarms, and take corrective actions as needed.\n5.  **Alarm and Notification System:** The system includes an alarm and notification system that alerts operators to any deviations or abnormalities in the industrial process. This system can send notifications to operators through various means, such as email, SMS, or voice calls.\n6.  **Data Storage and Reporting:** The system provides data storage and reporting capabilities, allowing operators to store historical data and generate reports on the industrial process. This feature is useful for analyzing trends, identifying areas for improvement, and making informed decisions.\n\nIn summary, the MAPROTECH X5000 Industrial Process Monitoring System is a comprehensive solution that integrates data acquisition, communication, processing, and presentation to provide real-time monitoring and analysis of industrial processes. Its key components work together to provide a robust and reliable monitoring system that helps operators optimize their processes and improve overall efficiency."
    text2 = "Based on the provided context, the key components of the MAPROTECH X5000 Industrial Process Monitoring System are:\n\n1. Advanced monitoring capabilities: The system can monitor various parameters such as sensor data and machine behavior to analyze and predict potential equipment failures.\n2. Predictive maintenance: The system uses advanced algorithms and machine learning techniques to analyze sensor data and machine behavior to predict potential equipment failures and schedule maintenance activities accordingly.\n3. Energy management: The system monitors energy consumption patterns and identifies opportunities for optimization, helping to reduce energy costs and improve sustainability.\n4. Quality control: The system uses advanced algorithms and machine learning techniques to detect even the slightest deviations from quality standards, ensuring consistent product quality and minimizing waste.\n5. Remote monitoring: The system allows authorized personnel to monitor and control production processes remotely, enabling timely interventions and efficient resource allocation.\n\nThese components work together to provide a comprehensive industrial process monitoring system that can:\n\n* Predict and prevent equipment failures, reducing unplanned downtime and extending asset lifespan\n* Optimize energy consumption, reducing energy costs and improving sustainability\n* Ensure consistent product quality, minimizing waste and improving overall efficiency\n* Enable remote monitoring and control, allowing for timely interventions and efficient resource allocation\n\nThe system's advanced monitoring capabilities, predictive analytics, and seamless integration enable it to provide a powerful competitive advantage in today's rapidly evolving industrial landscape."

    eval_obj.evaluate("mistral.mixtral-8x7b-instruct-v0:1",text1,text2,ground_truth)
    """
    

    #create clean-up scripts 

    #delete_all()