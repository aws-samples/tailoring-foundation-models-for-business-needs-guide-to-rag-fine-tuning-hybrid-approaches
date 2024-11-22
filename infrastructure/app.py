#!/usr/bin/env python3
import os

import aws_cdk as cdk
from constructs import DependencyGroup

from config import EnvSettings, KbConfig

from stacks.kb_role_stack import KbRoleStack
from stacks.oss_infra_stack import OpenSearchServerlessInfraStack
from stacks.kb_infra_stack import KbInfraStack
from stacks.s3_stack import S3Stack


app = cdk.App()

S3Stack(app, "S3Stack")

# create IAM role for RAG

KbRoleStack(app, "KbRoleStack")

# setup OSS
OpenSearchServerlessInfraStack(app, "OpenSearchServerlessInfraStack")

# # create Knowledgebase and datasource
KbInfraStack(app, "KbInfraStack")

app.synth()
