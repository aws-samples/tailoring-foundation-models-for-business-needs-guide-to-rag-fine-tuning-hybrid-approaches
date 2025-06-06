import os, json, boto3,re, json
from utils.bedrock import BedrockHandler
from utils.helpers import logger

class LLMEvaluator:
    """
    This module implements the LLM-as-Judge evaluation approach, using one LLM to evaluate
    the outputs of other LLMs. It specifically compares responses from three different 
    approaches: fine-tuning, RAG, and hybrid (finetuned model combined with rag) methods.
    """

    def __init__(self, bedrock_runtime):
        """
        Initialize the LLM Evaluatior class with required configurations.
        
        """
        self.bedrock_runtime = bedrock_runtime

    def evaluate(self,model_id, finetuning_text, rag_text, hybrid_text, ground_truth, prompt, pattern):
        """
        Args:
            model_id (str): The identifier of the Bedrock model to use as judge.
            finetuning_text (str): Output from the fine-tuned model approach.
            rag_text (str): Output from the RAG-based approach.
            hybrid_text (str): Output from the hybrid approach.
            ground_truth (str): The reference answer for evaluation.
            prompt (str): The prompt instructing the judge model how to evaluate.
            pattern (str): Regular expression pattern to extract scores from the judge's response.

        Returns:
            tuple: Three float values representing evaluation scores:
                - finetuning_score: Score for the fine-tuning approach
                - rag_score: Score for the RAG approach
                - hybrid_score: Score for the hybrid approach

        Raises:
            IndexError: If there's an error parsing scores from the judge's response.
            Exception: If the judge's response doesn't match the expected format.

        Note:
            The scoring pattern and scale should be clearly defined in the evaluation
            prompt to ensure consistent and meaningful results.
        """
        bedrock_handler = BedrockHandler(
            self.bedrock_runtime, model_id
        )

        message = [{
            "role": "user",
            "content": [{"text": f"Question: {prompt}"}],
        }]
        
        response = bedrock_handler.invoke_model(message)

        response_text = response['output']['message']['content'][0]['text']

        scores = re.search(pattern, response_text, re.DOTALL)

        if scores:
            json_string = scores.group(1).strip()

            if not json_string.strip().startswith('{'): #to fix bad formatted text
                json_string = '{' + json_string + '}'

            json_string = re.sub(r'\s+', ' ', json_string)  # normalize whitespace
            json_string = json_string.replace("'", '"')

            # Fix common JSON formatting issues
            json_string = re.sub(r'"}*\s*$', '"}', json_string)  # Fix extra quotes at the end
            json_string = re.sub(r'"\s*,\s*}', '"}', json_string)  # Fix trailing comma
            json_string = re.sub(r'(\d+)"(?=\s*[,}])', r'\1', json_string)  # Fix quoted numbers


            try:
                json_object = json.loads(json_string)

                # Extract scores
                finetuning_score = json_object.get('text1_score')
                rag_score = json_object.get('text2_score')
                hybrid_score = json_object.get('text3_score')

            except IndexError as e:
                logger.error("Error accessing the scores:", e)
                logger.error(f"Response text: {response_text}")
            except json.JSONDecodeError as e:
                logger.error("Error parsing JSON:", e)
                logger.error(f"JSON string: {json_string}")
                finetuning_score = rag_score = hybrid_score = -1


        else:
            logger.error("No matches found.")
            logger.error(f"Scores: {scores}")
            logger.error(f"Message: {message}")
            logger.error(f"Response: {response}")
            logger.error(f"Response text: {response_text}")

        return finetuning_score, rag_score, hybrid_score
