import os,re
import subprocess
from utils.helpers import logger

flag = "--require-approval never"
rag_dir = "./infrastructure"

# not executed now, assuming the user runs via commandline
def set_environment_variables():
    # Set your environment variables
    os.environ["AWS_REGION"] = "<region>"
    os.environ["AWS_ACCESS_KEY_ID"] = "<access-key>"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "<secret-key>"


def run_command(command, cwd=None):
    """Run a shell command, print output in real-time, and capture it."""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    output = []

    # Reading stdout line by line and printing it in real-time
    for line in iter(process.stdout.readline, b''):
        decoded_line = line.decode()
        print(decoded_line, end="")
        output.append(decoded_line)  # Store the stdout line in a list

    # Ensure all stdout is processed
    process.stdout.close()

    # Reading stderr line by line and printing it in real-time
    for line in iter(process.stderr.readline, b''):
        decoded_line = line.decode()
        print(decoded_line, end="")
        output.append(decoded_line)  # Store the stderr line in a list

    # Ensure all stderr is processed
    process.stderr.close()

    # Wait for the process to complete
    return_code = process.wait()

    # Check for any errors during command execution
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command, output="".join(output))

    return "".join(output)  # Return the combined output as a single string


def run_command_simpler(command, cwd=None):
    result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
    return result.stdout

def delete_all():
    run_command(f"cdk destroy --all {flag}", cwd=rag_dir)

def build_kb():
    
    # Run the prepare script
    logger.info("Running the prepare script...")
    run_command("./prepare.sh", cwd=rag_dir)

    # Bootstrap CDK
    logger.info("Bootstrapping CDK...")
    run_command("cdk bootstrap", cwd=rag_dir)
    
    # Synthesize CDK
    logger.info("Synthesizing CDK...")
    run_command("cdk synth", cwd=rag_dir)
    
    # Deploy specific stacks
    logger.info("Deploying S3Stack...")
    run_command(f"cdk deploy S3Stack {flag}", cwd=rag_dir)

    logger.info("Deploying KbRoleStack...")
    run_command(f"cdk deploy KbRoleStack {flag}", cwd=rag_dir)

    logger.info("Deploying OpenSearchServerlessInfraStack...")
    run_command(f"cdk deploy OpenSearchServerlessInfraStack {flag}", cwd=rag_dir)
    

    logger.info("Deploying KbInfraStack...")
    output = run_command(f"cdk deploy KbInfraStack {flag}", cwd=rag_dir)

    kb_id = re.search(r'KbInfraStack\.KnowledgeBaseId\s*=\s*([A-Z0-9]+)', output).group(1)

    logger.info("Infrastructure build complete.")
    return kb_id
