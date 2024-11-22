
from bleurt import score
import tensorflow as tf
import json
from evaluate import load
import numpy as np
import os, json, boto3
from src import llm_evaluator
from utils.helpers import load_json_file

from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class EvaluationResults:
    """Class to hold evaluation results"""
    bert_scores: Dict[str, List[float]]
    llm_evaluator_scores: Dict[str, List[Dict]]

class Evaluation:

    def __init__(self, bedrock_region: str, evaluator_models: dict, evaluator_prompt_template: str, score_pattern: str):
        """
        Initialize the Evaluation class with required configurations.
        
        Args:
            bedrock_region (str): AWS region for Bedrock.
            evaluator_models (dict): Model names an ids in Bedrock will be used as an evaluator
        """

        self.bedrock_runtime = boto3.client(
        service_name="bedrock-runtime", region_name=bedrock_region
        )
        self.evaluator_models = evaluator_models
        self.evaluator_prompt_template = evaluator_prompt_template

        self.llm_evaluator_obj = llm_evaluator.LLMEvaluator(
            self.bedrock_runtime,
        )
        self.score_pattern = score_pattern


    def calculate_bert(self, ground_truth: List[str], llm_generated: List[str]) -> Dict[str, List[float]]:
        """
        Calculate BERT scores for generated texts.
        
        Args:
            ground_truth: List of ground truth texts
            llm_generated: List of LLM generated texts
            
        Returns:
            Dictionary containing BERT scores
        """
        bertscore = load("bertscore")
        return bertscore.compute(
            predictions=llm_generated,
            references=ground_truth,
            lang="en"
        )


    def calculate_llm_evaluator(self,
                              ground_truth: str,
                              llm_response_finetuning: str,
                              llm_response_rag: str,
                              llm_response_hybrid: str) -> Dict[str, Dict]:
        """
        Calculate LLM evaluator scores for different approaches.
        
        Args:
            ground_truth: Ground truth text
            llm_response_finetuning: Response from finetuning approach
            llm_response_rag: Response from RAG approach
            llm_response_hybrid: Response from hybrid approach
            
        Returns:
            Dictionary containing scores for each approach
        """
        scores = {
            "finetuning" : {},
            "rag": {},
            "hybrid": {}
        }
        
        finetuning_sum = 0
        rag_sum = 0 
        hybrid_sum = 0

        prompt = self.evaluator_prompt_template.format(
            ground_truth=ground_truth,
            finetuning_text=llm_response_finetuning,
            rag_text=llm_response_rag,
            hybrid_text=llm_response_hybrid
        )

        for model_name in self.evaluator_models.keys():
            model_id = self.evaluator_models[model_name]

            score_finetuning, score_rag, score_hybrid = self.llm_evaluator_obj.evaluate(
                model_id,
                llm_response_finetuning,
                llm_response_rag,
                llm_response_hybrid,
                ground_truth,
                prompt,
                self.score_pattern
            )

            scores['finetuning'][model_name] = score_finetuning 
            scores['rag'][model_name] = score_rag 
            scores['hybrid'][model_name] = score_hybrid 
            finetuning_sum += score_finetuning
            rag_sum += score_rag
            hybrid_sum += score_hybrid

        sample_count = len(scores['finetuning'].keys())
        scores['finetuning']['llm_evaluator_score'] = finetuning_sum/sample_count
        scores['rag']['llm_evaluator_score'] = rag_sum/sample_count
        scores['hybrid']['llm_evaluator_score'] = hybrid_sum/sample_count

        return scores

    def calculate_scores(self,
                        finetuning_file: str,
                        rag_file: str,
                        hybrid_file: str) -> Tuple[List[float], List[float], List[float], List[Dict], List[Dict], List[Dict]]:
        """
        Calculate evaluation scores for all approaches.
        
        Args:
            finetuning_file: Path to finetuning results
            rag_file: Path to RAG results
            hybrid_file: Path to hybrid results
            
        Returns:
            Tuple containing BERT scores and LLM evaluator scores for each approach
        """
       
       # Load data
        finetuning_data = load_json_file(finetuning_file)
        rag_data = load_json_file(rag_file)
        hybrid_data = load_json_file(hybrid_file)

        ground_truth = [entry["ground_truth"] for entry in finetuning_data] # ground truth is same for each method

        # Retrieving llm generated texts
        llm_response_finetuning = [entry["llm_response"] for entry in finetuning_data]
        llm_response_rag = [entry["llm_response"] for entry in rag_data]
        llm_response_hybrid = [entry["llm_response"] for entry in hybrid_data]

                # Calculate BERT scores
        bert_scores = {
            'finetuning': self.calculate_bert(ground_truth, llm_response_finetuning)['f1'],
            'rag': self.calculate_bert(ground_truth, llm_response_rag)['f1'],
            'hybrid': self.calculate_bert(ground_truth, llm_response_hybrid)['f1']
        }

        # Calculate LLM evaluator scores
        llm_evaluator_scores = {
            'finetuning': [],
            'rag': [],
            'hybrid': []
        }
        
        for i in range(len(ground_truth)):
            scores = self.calculate_llm_evaluator(
                ground_truth[i],
                llm_response_finetuning[i],
                llm_response_rag[i],
                llm_response_hybrid[i]
            )
            
            llm_evaluator_scores['finetuning'].append(scores['finetuning'])
            llm_evaluator_scores['rag'].append(scores['rag'])
            llm_evaluator_scores['hybrid'].append(scores['hybrid'])

        return EvaluationResults(bert_scores, llm_evaluator_scores)



    def save_scores(self, results: EvaluationResults, finetuning_file: str, rag_file: str, hybrid_file: str) -> None:
        """
        Save evaluation scores to respective files.
        
        Args:
            results: EvaluationResults containing scores
            finetuning_file: Path to finetuning results
            rag_file: Path to RAG results
            hybrid_file: Path to hybrid results
        """
        files_data = {
            'finetuning': (finetuning_file, results.bert_scores['finetuning'], results.llm_evaluator_scores['finetuning']),
            'rag': (rag_file, results.bert_scores['rag'], results.llm_evaluator_scores['rag']),
            'hybrid': (hybrid_file, results.bert_scores['hybrid'], results.llm_evaluator_scores['hybrid'])
        }

        for approach, (file_path, bert_score, llm_scores) in files_data.items():
            data = load_json_file(file_path)
            
            for i, entry in enumerate(data):
                entry["bert_score"] = bert_score[i]
                entry.update(llm_scores[i])
            
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
    
        
    def calculate_aggregated_scores(self, finetuning_file: str, rag_file: str, hybrid_file: str) -> None:
        """
        Calculate and print aggregated scores for all approaches.
        
        Args:
            finetuning_file: Path to finetuning results
            rag_file: Path to RAG results
            hybrid_file: Path to hybrid results
        """
        files_data = {
            'finetuning': finetuning_file,
            'rag': rag_file,
            'hybrid': hybrid_file
        }

        for approach, file_path in files_data.items():
            data = load_json_file(file_path)
            
            bert_score = np.mean([sample['bert_score'] for sample in data])
            llm_eval_score = np.mean([sample['llm_evaluator_score'] for sample in data])
            
            print(f"\n{approach.upper()} Scores:")
            print(f"BERT Score: {bert_score:.4f}")
            print(f"LLM Evaluator Score: {llm_eval_score:.4f}")
            print("--------------------")


