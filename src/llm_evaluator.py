import os, json, boto3,re
from utils.bedrock import BedrockHandler

class LLMEvaluator:
    """
    A class to implement LLM Evaluator using Bedrock.
    """

    def __init__(self, bedrock_runtime):
        """
        Initialize the LLM Evaluatior class with required configurations.
        
        """
        self.bedrock_runtime = bedrock_runtime

    def evaluate(self,model_id, finetuning_text, rag_text, hybrid_text, ground_truth):
        bedrock_handler = BedrockHandler(
            self.bedrock_runtime, model_id
        )
        #TODO: Read from config
        prompt = (
            "You are an AI assistant to evaluate different AI-generated texts under consideration of the ground truth. "
            "I will provide you a ground truth followed by different AI-generated answers for questions on a product catalog. "
            "Your score range should be in the range 0-1. Evaluate the accuracy and quality of the LLM responses using the following criteria:\n\n"
            "1. Correctness: Does the response match the ground truth answer? Are the facts and details aligned with what’s provided in the ground truth?\n"
            "2. Completeness: Does the response include all relevant points found in the ground truth answer? Are there any omissions or missing details?\n"
            "3. Clarity and Readability: Is the response clear and easy to understand? Does it convey information in a way that would be understandable to the user?\n"
            "4. No Hallucinations: Does the response avoid introducing any information that is not present in the ground truth? Ensure that no additional or fabricated details are present.\n\n"
            f"Ground Truth: {ground_truth}\n\n"
            f"Text 1: {finetuning_text}\n\n"
            f"Text 2: {rag_text}\n\n"
            f"Text 3: {hybrid_text}\n\n"
            "Provide your evaluation score in this format for Text1, Text2 and Text3. Make sure that you only provide the scores without explanation:\n"
            "Text1 Score: [score]\n"
            "Text2 Score: [score]\n"
            "Text3 Score: [score]\n"
        )

        message = [{
            "role": "user",
            "content": [{"text": f"Question: {prompt}"}],
        }]
        
        response = bedrock_handler.invoke_model(message)
        response_text = response['output']['message']['content'][0]['text']

        # Regex pattern to match integer or decimal scores
        pattern = r"Text1 Score: (\d+\.?\d*)[^\n]*\nText2 Score: (\d+\.?\d*)[^\n]*\nText3 Score: (\d+\.?\d*)[^\n]*"

        scores = re.search(pattern, response_text)

        if scores:
            try:
                finetuning_score = float(scores.group(1))
                rag_score = float(scores.group(2))
                hybrid_score = float(scores.group(3))

            except IndexError as e:
                print("Error accessing the scores:", e)
                print(f"Response text: {response_text}")

        else:
            print("No matches found.")
            print(f"Response text: {response_text}")

        return finetuning_score, rag_score, hybrid_score
