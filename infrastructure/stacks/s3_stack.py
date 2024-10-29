from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_iam as iam,
    aws_ssm as ssm,
    RemovalPolicy,
)
from constructs import Construct
from config import EnvSettings, DsConfig


region = EnvSettings.ACCOUNT_REGION
account_id = EnvSettings.ACCOUNT_ID
bucket_name = DsConfig.S3_BUCKET_NAME

class S3Stack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create the S3 bucket
        bucket = s3.Bucket(
            self,
            "Rag-Finetuning-Comparison",
            bucket_name=bucket_name, 
            versioned=True, 
            removal_policy=RemovalPolicy.DESTROY,  # Automatically delete the bucket when the stack is deleted
            auto_delete_objects=True  # Automatically delete objects when the bucket is deleted
        )

        # Create an IAM role with S3 access
        self.s3_role = iam.Role(
            self,
            "S3Role",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal(
                "bedrock.amazonaws.com",
                conditions={
                    "StringEquals": {"aws:SourceAccount": account_id},
                    "ArnLike": {"aws:SourceArn": f"arn:aws:s3:::{bucket.bucket_name}/*"}
                },
                ),
                iam.ServicePrincipal(
                "sagemaker.amazonaws.com",
                conditions={
                    "StringEquals": {"aws:SourceAccount": account_id},
                    "ArnLike": {"aws:SourceArn": f"arn:aws:s3:::{bucket.bucket_name}/*"}
                },
                )
            ),
            inline_policies={
                "S3Policy": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            sid="BedrockInvokeModelStatement",
                            effect=iam.Effect.ALLOW,
                            actions=[
                                    "s3:PutObject",      
                                    "s3:PutObjectAcl",    
                                    "s3:GetObject",       
                                    "s3:DeleteObject" ,
                                    "s3:ListBucket"  
                                ],
                                resources=[f"{bucket.bucket_arn}/*"],
                        )
                    ]
                )
            }
        )

        # create an SSM parameters which store export values
        ssm.StringParameter(self, 's3RoleArn',
                            parameter_name="/e2e-rag/s3RoleArn",
                            string_value=self.s3_role.role_arn)