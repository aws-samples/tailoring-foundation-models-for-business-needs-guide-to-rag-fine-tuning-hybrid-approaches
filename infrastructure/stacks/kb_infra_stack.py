from constructs import Construct
import json 
import aws_cdk as core
from aws_cdk import (
    Duration,
    Stack,
    aws_iam as iam,
    aws_sqs as sqs,
    aws_sns as sns,
    aws_lambda as lambda_,
    aws_sns_subscriptions as subs,
    aws_logs as logs,
    aws_ssm as ssm,
    aws_events as events,
    # aws_bedrock as bedrock,
    aws_s3 as s3,
)

from aws_cdk import aws_bedrock as bedrock
from aws_cdk import aws_s3_notifications as s3_notifications

from aws_cdk.aws_bedrock import (
  CfnKnowledgeBase,
  CfnDataSource
)

from config import EnvSettings, KbConfig, DsConfig, OpenSearchServerlessConfig
from aws_cdk import custom_resources as cr
from aws_cdk import CfnOutput

region = EnvSettings.ACCOUNT_REGION
account_id = EnvSettings.ACCOUNT_ID

collectionName= OpenSearchServerlessConfig.COLLECTION_NAME
indexName= OpenSearchServerlessConfig.INDEX_NAME


embeddingModelId= KbConfig.EMBEDDING_MODEL_ID
max_tokens = KbConfig.MAX_TOKENS
overlap_percentage = KbConfig.OVERLAP_PERCENTAGE
kb_name = KbConfig.KB_NAME


embeddingModelArn = f"arn:aws:bedrock:{region}::foundation-model/{embeddingModelId}"
bucket_name = DsConfig.S3_BUCKET_NAME
s3_bucket_arn = f"arn:aws:s3:::{bucket_name}"

kb_folder_name = DsConfig.KB_DATA_FOLDER


class KbInfraStack(Stack):

  def __init__(self, scope: Construct, construct_id: str)-> None:
        super().__init__(scope, construct_id)

        self.kbRoleArn = ssm.StringParameter.from_string_parameter_attributes(self, "kbRoleArn",
                        parameter_name="/e2e-rag/kbRoleArn").string_value
        print("kbRoleArn: " + self.kbRoleArn)
        self.collectionArn = ssm.StringParameter.from_string_parameter_attributes(self, "collectionArn",
                        parameter_name="/e2e-rag/collectionArn").string_value
        print("collectionArn: " + self.collectionArn)

        #   Create Knowledgebase
        self.knowledge_base = self.create_knowledge_base()
        self.data_source = self.create_data_source(self.knowledge_base)
        self.ingest_lambda = self.create_ingest_lambda(self.knowledge_base, self.data_source)
        self.sync_data_source(self.ingest_lambda)

        CfnOutput(self, "KnowledgeBaseId", value=self.knowledge_base.attr_knowledge_base_id)
        CfnOutput(self, "DataSourceId", value=self.data_source.attr_data_source_id) 


    
  def create_knowledge_base(self) -> CfnKnowledgeBase:
    return CfnKnowledgeBase(
        self, 
        'e2eRagKB',
        knowledge_base_configuration=CfnKnowledgeBase.KnowledgeBaseConfigurationProperty(
        type="VECTOR",
        vector_knowledge_base_configuration=CfnKnowledgeBase.VectorKnowledgeBaseConfigurationProperty(
            embedding_model_arn=embeddingModelArn
        )
        ),
        name=kb_name,
        role_arn=self.kbRoleArn,
        # the properties below are optional
        description='e2eRAG Knowledge base',
        storage_configuration=CfnKnowledgeBase.StorageConfigurationProperty(
          type="OPENSEARCH_SERVERLESS",
          # the properties below are optional
            opensearch_serverless_configuration=bedrock.CfnKnowledgeBase.OpenSearchServerlessConfigurationProperty(
                collection_arn=self.collectionArn,
                field_mapping=bedrock.CfnKnowledgeBase.OpenSearchServerlessFieldMappingProperty(
                    metadata_field="AMAZON_BEDROCK_METADATA",
                    text_field="AMAZON_BEDROCK_TEXT_CHUNK",
                    vector_field="bedrock-knowledge-base-default-vector"
                ),
                vector_index_name=indexName
            )
        )
      )
  
  def create_data_source(self, knowledge_base) -> CfnDataSource:
    kbid = knowledge_base.attr_knowledge_base_id
    chunking_strategy = KbConfig.CHUNKING_STRATEGY
    if chunking_strategy == "Fixed-size chunking":
      vector_ingestion_config_variable = bedrock.CfnDataSource.VectorIngestionConfigurationProperty(
                chunking_configuration=bedrock.CfnDataSource.ChunkingConfigurationProperty(
                    chunking_strategy="FIXED_SIZE",
                    # the properties below are optional
                    fixed_size_chunking_configuration=bedrock.CfnDataSource.FixedSizeChunkingConfigurationProperty(
                        max_tokens=max_tokens,
                        overlap_percentage=overlap_percentage
                    )
                )
            )
    elif chunking_strategy == "Default chunking":
      vector_ingestion_config_variable = bedrock.CfnDataSource.VectorIngestionConfigurationProperty(
                chunking_configuration=bedrock.CfnDataSource.ChunkingConfigurationProperty(
                    chunking_strategy="FIXED_SIZE",
                    # the properties below are optional
                    fixed_size_chunking_configuration=bedrock.CfnDataSource.FixedSizeChunkingConfigurationProperty(
                        max_tokens=300,
                        overlap_percentage=20
                    )
                )
            )
    else:
      vector_ingestion_config_variable = bedrock.CfnDataSource.VectorIngestionConfigurationProperty(
                chunking_configuration=bedrock.CfnDataSource.ChunkingConfigurationProperty(
                    chunking_strategy= "NONE"
                )
            )
    return CfnDataSource(self, "e2eRagDataSource",
    data_source_configuration=CfnDataSource.DataSourceConfigurationProperty(
        s3_configuration=CfnDataSource.S3DataSourceConfigurationProperty(
            bucket_arn=s3_bucket_arn,
            # the properties below are optional
            # inclusion_prefixes=["inclusionPrefixes"]
            inclusion_prefixes=[kb_folder_name]
        ),
        type="S3"
    ),
    knowledge_base_id= kbid,
    name="e2eRAGDataSource",

    # the properties below are optional
    description="e2eRAG DataSource",
    
    vector_ingestion_configuration=vector_ingestion_config_variable
    )

  def create_ingest_lambda(self, knowledge_base, data_source) -> lambda_:
    ingest_lambda= lambda_.Function(
        self,
        "IngestionJob",
        runtime=lambda_.Runtime.PYTHON_3_10,
        handler="ingestJobLambda.lambda_handler",
        code=lambda_.Code.from_asset("./src/IngestJob"),
        timeout=Duration.minutes(15),
        environment=dict(
            KNOWLEDGE_BASE_ID=knowledge_base.attr_knowledge_base_id,
            DATA_SOURCE_ID=data_source.attr_data_source_id,
        )
    )
    # lambda_ingestion_job.add_event_source(s3_put_event_source)

    ingest_lambda.add_to_role_policy(iam.PolicyStatement(
        actions=["bedrock:StartIngestionJob"],
        resources=[knowledge_base.attr_knowledge_base_arn]
    ))
    return ingest_lambda

  def sync_data_source(self,ingest_lambda):
    bucket = s3.Bucket.from_bucket_name(
            self,
            "ProductCatalogBucket",
            bucket_name, #"product-catalog-bucket-nvirginia"
        )
    bucket.grant_read(ingest_lambda)
        
    # Add the S3 event notification for object creation
    bucket.add_event_notification(
        s3.EventType.OBJECT_CREATED,
        s3_notifications.LambdaDestination(ingest_lambda),
        s3.NotificationKeyFilter(prefix=kb_folder_name)  # Only triggers for the specified prefix
    )
    
    # Grant the Lambda function permission to start the ingestion job in Bedrock
    ingest_lambda.add_to_role_policy(iam.PolicyStatement(
        actions=["bedrock:StartIngestionJob"],
        resources=[f"arn:aws:bedrock:{region}:{account_id}:knowledge-base/{kb_name}"]
    ))
