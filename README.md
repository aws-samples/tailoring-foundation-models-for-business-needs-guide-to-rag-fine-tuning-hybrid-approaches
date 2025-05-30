
# Tailoring foundation models for your business needs: A comprehensive guide to RAG, fine-tuning, and hybrid approaches

This code sample provides a framework to evaluate and compare different FM customization strategies. It helps to determine the most effective approach for specific use cases by analyzing performance, cost and latency across **RAG**, **Fine-tuning**, and **Hybrid** (RAG on top of fine-tuning) methods. We have provided a synthetic data to experiment with, you shall replace it with your own data related to your use case, then you can run the evaluation by following the below instructions.

---

## Prerequisites
- Python >=3.9
- AWS Account with appropriate permissions
- Docker

---

## Installation

1. ### **Clone the repository**  
   ```bash
   git clone https://github.com/aws-samples/tailoring-foundation-models-for-business-needs-guide-to-rag-fine-tuning-hybrid-approaches.git
   ```

2. ### **Set up Python environment**  
   ```bash
   cd tailoring-foundation-models-for-business-needs-guide-to-rag-fine-tuning-hybrid-approaches/
   export PYTHONPATH=$(pwd)
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

3. ### **Install the required Python packages**  
   ```bash
   pip install -r requirements.txt
   ```

4. ### **Update config files**  
    Make the respective changes in the **config.py** file 
   ##### 1. Environment Settings
   Update the `EnvSettings` class with your AWS account information:
   ```python
   ACCOUNT_ID = "your-aws-account-id"    # Required: Your AWS account ID
   ACCOUNT_REGION = "your-region"        # Required: AWS region (e.g., "us-east-1"), Finetuning functioning should be available in the region you choose
   RAG_PROJ_NAME = "your-project-name"   # Required: Choose a project name
   ```
   ##### 2. Knowledge Base Configuration
   Update the `KbConfig` class, 
   ```
   Choose chunking strategy::: This is not implemented yet so will be updated later.
   ```
   ##### 3. Data Source Configuration
   Update the `DsConfig` class, 
   ```python
   S3_BUCKET_NAME = f"rag-finetuning-comparison-{ACCOUNT_ID}" #f"product-catalog-bucket-nvirginia" # Optional: Change this to the S3 bucket where your data is stored
   KB_DATA_FOLDER = f"kb-data" #Optional: Change this to the folder where your kb data is stored (under the S3 Bucket you have choosed previously)
    ```

    ##### 4. RAG Configuration
    Update the `RAGConfig` class, 
    ```python
    NUMBER_OF_RESULTS = 3  # Optional: Number of documents will be used as context in RAG

    ```

    ##### 5. Finetuning Configuration
    Update the `FinetuningConfig` class, 
    ```python
    NUM_EPOCH = 3  # Optional: Adjust if needed, if you are working with a small dataset, higher values might cause overfitting.
    ```

    ##### 6. Evaluation Configuration
    Update the `EvaluationConfig` class, 
    ```python
    MODELS_EVAL = {
        "mistral_8_7b": "mistral.mixtral-8x7b-instruct-v0:1",
        "command_r_plus": "cohere.command-r-plus-v1:0",
        "claude3_haiku": "anthropic.claude-3-haiku-20240307-v1:0"
    } # Optional: These models will be used as an evaluators, you can delete or add models from Bedrock. 
    
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
    )  # Optional: This prompt will be used in evaluation, it generates scores for each of three approaches. 
   
    SCORE_PATTERN = r"Text1 Score: (\d+\.?\d*)[^\n]*\nText2 Score: (\d+\.?\d*)[^\n]*\nText3 Score: (\d+\.?\d*)[^\n]*" # Optional: It is used to capture generated 3 different scores, if you use other evaluator models, it might need some adjustments. 
    ```

    ##### 7. Templates Configuration
    Update the `Templates` class, 
    ```python
    FINETUNING_TEMPLATE # Required: Modify the prompt part accordingly for your usecase.
    HYBRID_TEMPLATE # Required: Modify the prompt part accordingly for your usecase.
    RAG_TEMPLATE # Required: Modify the prompt part accordingly for your usecase.
    ```


## Infrastructure Deployment
To run this solution, you'll need several AWS services working together. We automated this setup using AWS Cloud Development Kit (CDK). It consist of 4 different stacks, 
- S3Stack
    * Creates an S3 bucket to store our product catalog dataset
    * Sets up proper security configurations for bucket access.
- KBRoleStack
    * Creates IAM roles and policies
    * Grants permissions for Knowledge Base access and OpenSearch Serverless operations.
- OpenSearchServerlessInfraStack
    * Sets up Amazon OpenSearch Serverless 
    * Creates vector store for document retrieval
- KBInfraStack
    * Creates the Knowledge Base 
    * Sets up data source connections to S3
    * Synchronization between S3 and Knowledge Base 

    You can deploy all of the stacks by running commands below. Make sure you are also running docker.   

```bash
cd infrastructure 
./prepare.sh
cdk bootstrap
cdk synth
cdk deploy --all
cd ..
```

---

## Dataset Preparation

### Required Datasets
1. **Knowledge Base Dataset**  
   Data for the RAG approach, consisting of one or more text documents.
   
2. **Training Dataset**  
   Question-Answer pairs related to the knowledge base, in JSON format for training.
   
3. **Testing Dataset**  
   Question-Answer pairs related to the knowledge base, in JSON format for evaluation.

### Dataset Placement
Organize your datasets as follows:

```
data/
├── kb-data/    # Knowledge base data for RAG
├── train/      # Training data for fine-tuning
└── test/       # Testing data for evaluation
```

Ensure your datasets are in the correct format.  
For instruction fine-tuning with **Llama 3.1 8B Instruct**, training and testing data should follow the **Question-Answer pairs** format. If you intend to use a different fine-tuning approach, re-implement the fine-tuning logic and prepare the dataset accordingly.

---

## Usage

### Request model access to Amazon Bedrock models
Use [this guide](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access-modify.html) to request access to the follwoing Bedrock models:
1. Llama 3 8B Instruct -> Used for RAG
2. Amazon Titan Text Embeddings V2 -> Used for RAG
3. Mixtral 8x7B Instruct -> Used for evaluating the results
4. Command R+  -> Used for evaluating the results
5. Claude 3 Haiku  -> Used for evaluating the results



### Request quota increase for Amazon SageMaker AI endpoint usage
In your AWS console:
1. navigate to `Service Quotas` service
2. On the left handside select `AWS Services`
3. Select Amazon SageMaker
4. Then select `ml.g5.12xlarge for endpoint usage` 
5. Press on `Request Increase at account level`, and set the quota value to 1. This should get automatically approved.


### Run the main script
Make sure that you are in the root directory of the repository, then use the following command to run the main script:
```bash
python main.py
```

---

## Clean-up

### 1. Infrastructure Clean-up
To delete all resources created during the infrastructure setup:

```bash
cd infrastructure/
cdk destroy --all
```

### 2. Endpoint Management

To delete the SageMaker endpoint for the finetuned model:
1. Access the **Amazon SageMaker AI Console**.
2. Navigate to **Inference → Endpoints**.
3. Select and delete the desired endpoint. The endpoint name starts with `llama-3-1-8b-instruct-****`.

--- 
