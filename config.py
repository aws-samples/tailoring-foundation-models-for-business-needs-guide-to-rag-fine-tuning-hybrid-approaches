'''
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
This source code is subject to the terms found in the AWS Enterprise Customer Agreement.
NOTE: If deploying to production, set this to true.
 - If this is set to true, all properties from the Prod_Props class will be used
 - If this is set to false, all the properties from the Dev_Props class will be used
'''


EMBEDDING_MODEL_IDs = ["amazon.titan-embed-text-v2:0"]
CHUNKING_STRATEGIES = {0:"Default chunking",1:"Fixed-size chunking", 2:"No chunking"}

#Sagemaker
MODELS_FINETUNING = {
        "llama3_8b_instruct": "meta-textgeneration-llama-3-1-8b-instruct"
    }
#Bedrock
MODELS_RAG = {
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
    NUMBER_OF_RESULTS = 3  # TODO: You can try different context count for RAG

class FinetuningConfig:
    MODEL_NAME = "llama3_8b_instruct"
    MODEL_ID = MODELS_FINETUNING["llama3_8b_instruct"] # TODO: Choose the Bedrock Model option (claude-v2, command-light, llama3_8b_instruct)
    METHOD = "domain_adaptation" # TODO: Choose Finetuning method. eg:"domain_adaptation", "instruction_finetuning"
    NUM_EPOCH = 8 # TODO: Adjust if needed, default is 5

class EvaluationConfig:
    MODELS_EVAL = {
        "mistral_8_7b": "mistral.mixtral-8x7b-instruct-v0:1",
        "command_r_plus": "cohere.command-r-plus-v1:0",
        "claude3_haiku": "anthropic.claude-3-haiku-20240307-v1:0"
    } #TODO: Add or extract bedrock model ids if necessary

    PROMPT_TEMPLATE = (
        "You are an AI assistant to evaluate different AI-generated texts under consideration of the ground truth. "
        "I will provide you a ground truth followed by different AI-generated answers for questions on a product catalog. "
        "Your score range should be in the range 0-1. Evaluate the accuracy and quality of the LLM responses using the following criteria:\n\n"
        "1. Correctness: Does the response match the ground truth answer? Are the facts and details aligned with what's provided in the ground truth?\n"
        "2. Completeness: Does the response include all relevant points found in the ground truth answer? Are there any omissions or missing details?\n"
        "3. Clarity and Readability: Is the response clear and easy to understand? Does it convey information in a way that would be understandable to the user?\n"
        "4. No Hallucinations: Does the response avoid introducing any information that is not present in the ground truth? Ensure that no additional or fabricated details are present.\n\n"
        "Ground Truth: {ground_truth}\n\n"
        "Text 1: {finetuning_text}\n\n"
        "Text 2: {rag_text}\n\n"
        "Text 3: {hybrid_text}\n\n"
        "Provide your evaluation score in this format for Text1, Text2 and Text3. Make sure that you only provide the scores without explanation:\n"
        "Text1 Score: [score]\n"
        "Text2 Score: [score]\n"
        "Text3 Score: [score]\n"
    )  #TODO: Modify it according to your usecase

    SCORE_PATTERN = r"Text1 Score: (\d+\.?\d*)[^\n]*\nText2 Score: (\d+\.?\d*)[^\n]*\nText3 Score: (\d+\.?\d*)[^\n]*" #TODO: Changes might needed for different prompt templates



class Templates:
    FINETUNING_TEMPLATE = {
        "prompt": (
            "You are a helpful AI assistant. You are intelligent and concise."
            " Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            " If you don't know the answer, just say, \"I don't know.\" Don't make anything up."
            " ### Instruction:\n Answer the customer's question: {question}\n\n"
        ),
        "completion": "{answer}",
    } #TODO: Modify it according to your usecase/model you are using

    HYBRID_TEMPLATE = {
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
    } #TODO: Modify it according to your usecase/model you are using

    RAG_TEMPLATE = {
        "prompt": (
            "You are a helpful AI assistant. You are intelligent and concise. Below is an instruction that describes a task, "
            "paired with an input that provides further context.\n"
            "Write a response that appropriately completes the request.\n\n"
            "If you don't know the answer, just say, \"I don't know.\" Don't make anything up.\n"
            "### Instruction:\n"
            "Answer the customer's question: {question}\n\n"
            "### Input:\n{context}\n\n"
        )
    } #TODO: Modify it according to your usecase/model you are using
