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

s3_stack = S3Stack(app, "S3Stack")

# create IAM role for RAG

kb_role_stack = KbRoleStack(app, "KbRoleStack")
kb_role_stack.add_dependency(s3_stack)

# setup OSS
oss_stack = OpenSearchServerlessInfraStack(app, "OpenSearchServerlessInfraStack")
oss_stack.add_dependency(kb_role_stack)

# create Knowledgebase and Datasource
kb_infra_stack = KbInfraStack(app, "KbInfraStack")
kb_infra_stack.add_dependency(oss_stack)


app.synth()
