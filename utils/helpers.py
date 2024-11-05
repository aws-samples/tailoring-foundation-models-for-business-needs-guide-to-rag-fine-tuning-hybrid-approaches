import logging, boto3, os, json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Get a shared logger instance
logger = logging.getLogger("app_logger")


def upload_data_S3(s3_client, data_folder_path, kb_data_folder, bucket_name):
    for file_name in os.listdir(kb_data_folder):
        file_path = os.path.join(kb_data_folder, file_name)
        if os.path.isfile(file_path):  # Check if it's a file (skip directories)
            try:
                # Upload the file to S3
                s3_key = f"kb-data/{file_name}"
                s3_client.upload_file(file_path, bucket_name, s3_key)
                print(f"Uploaded {file_name} to s3://{bucket_name}/kb-data/{file_name}")
            except Exception as e:
                print(f"Failed to upload {file_name}: {str(e)}")
        else:
            print("WARNING! There is nothing to upload!!")

def json_to_jsonl(json_file_path, output_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    with open(output_file_path, 'w') as outfile:
        for record in data:
            json.dump(record, outfile)
            outfile.write('\n') 

def template_and_predict(predictor, template, question, context, ground_truth, input_output_demarkation_key="\n\n### Response:\n"):
    prompt = template["prompt"]
    inputs = prompt.format(question=question, context=context)
    inputs += input_output_demarkation_key
    payload = {"inputs": inputs, "parameters": {"max_new_tokens": 4096}}
    for trial_count in range(0,5):
        try:
            response = predictor.predict(payload)
            return inputs, ground_truth, response
        except Exception as e:
            trial_count += 1
            print(f"Trial #{trial_count}, Error in template_and_predict: {e}")
            continue
    return inputs, ground_truth, 'Error!'
    


def update_trust_relationship(iam_client, role_name, principal_service, actions=None):
    """
    Updates the trust relationship of an IAM role.

    Parameters:
    - iam_client: The Boto3 IAM client.
    - role_name: Name of the IAM role to update.
    - principal_service: The service that will be allowed to assume the role (e.g., 'sagemaker.amazonaws.com').
    - actions: A list of actions to allow. Defaults to ['sts:AssumeRole'] if not provided.
    """
    if actions is None:
        actions = ["sts:AssumeRole"]
    
    trust_policy = {
        "Version": "2012-10-17", # update if needed in the future
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": { "Service": principal_service },
                "Action": actions
            }
        ]
    }

    iam_client.update_assume_role_policy(RoleName=role_name, PolicyDocument=json.dumps(trust_policy))

