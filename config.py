'''
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
This source code is subject to the terms found in the AWS Enterprise Customer Agreement.
NOTE: If deploying to production, set this to true.
 - If this is set to true, all properties from the Prod_Props class will be used
 - If this is set to false, all the properties from the Dev_Props class will be used
'''


EMBEDDING_MODEL_IDs = ["amazon.titan-embed-text-v2:0"]
CHUNKING_STRATEGIES = {0:"Default chunking",1:"Fixed-size chunking", 2:"No chunking"}
# TODO: model ids are differ from finetuning and rag. Make a nicer solution for handling this.
MODELS_FINETUNING = {
        #"claude-v2" : "anthropic.claude-v2", # check the id
        #"command-light": "cohere.command-light-text-v14", # check the id 
        "llama3_8b_instruct": "meta-textgeneration-llama-3-1-8b-instruct"
    }

MODELS_RAG = {
        "claude-v2" : "anthropic.claude-v2", # check the id
        "command-light": "cohere.command-light-text-v14", # check the id 
        "llama3_8b_instruct": "meta.llama3-8b-instruct-v1:0"
    }

class EnvSettings:
    # General params
    ACCOUNT_ID =  "339712995635" # TODO: Change this to your account
    ACCOUNT_REGION = "us-east-1" # TODO: Change this to your region
    RAG_PROJ_NAME = "rag-trial" # TODO: Change this to any name of your choice

class KbConfig:
    KB_ROLE_NAME = f"{EnvSettings.RAG_PROJ_NAME}-kb-role"
    KB_NAME = "docKnowledgeBase" #'docKnowledgeBase'
    EMBEDDING_MODEL_ID = EMBEDDING_MODEL_IDs[0]
    CHUNKING_STRATEGY = CHUNKING_STRATEGIES[1] # TODO: Choose the Chunking option 0,1,2
    MAX_TOKENS = 512 # TODO: Change this value accordingly if you choose "FIXED_SIZE" chunk strategy
    OVERLAP_PERCENTAGE = 20 # TODO: Change this value accordingly

class DsConfig:
    S3_BUCKET_NAME = f"rag-finetuning-comparison" #f"product-catalog-bucket-nvirginia" # TODO: Change this to the S3 bucket where your data is stored
    KB_DATA_FOLDER = f"kb-data" #TODO: Change this to the folder where your kb data is stored (under the S3 Bucket you have choosed previously)

class OpenSearchServerlessConfig:
    COLLECTION_NAME = f"{EnvSettings.RAG_PROJ_NAME}-kb-collection"
    INDEX_NAME = f"{EnvSettings.RAG_PROJ_NAME}-kb-index"

class RAGConfig:
    MODEL_NAME = "llama3_8b_instruct"
    MODEL_ID = MODELS_RAG["llama3_8b_instruct"]
    NUMBER_OF_RESULTS = 5  # TODO:

class FinetuningConfig:
    MODEL_NAME = "llama3_8b_instruct"
    MODEL_ID = MODELS_FINETUNING["llama3_8b_instruct"] # TODO: Choose the Bedrock Model option (claude-v2, command-light, llama3_8b_instruct)
    METHOD = "instruction_finetuning" # TODO: Choose Finetuning method. eg: "instruction_finetuning", "domain_adaptation" 


    