
from bleurt import score
import tensorflow as tf
import json
from evaluate import load
import numpy as np
import os, json, boto3
from src import llm_evaluator

class Evaluation:

    def __init__(self, bedrock_region: str, evaluator_models: dict):
        """
        Initialize the Evaluation class with required configurations.
        
        Args:
            bedrock_region (str): AWS region for Bedrock.
        """

        self.bedrock_runtime = boto3.client(
        service_name="bedrock-runtime", region_name=bedrock_region
        )
        self.evaluator_models = evaluator_models

        self.llm_evaluator_obj = llm_evaluator.LLMEvaluator(self.bedrock_runtime)


    def calculate_bert(self,ground_truth,llm_generated):
        bertscore = load("bertscore")
        scores = bertscore.compute(predictions=llm_generated, references=ground_truth, lang="en")
        return scores

    def calculate_llm_evaluator(self, ground_truth,llm_response_finetuning,llm_response_rag, llm_response_hybrid):
        scores = {
            "finetuning" : {},
            "rag": {},
            "hybrid": {}
        }
        finetuning_sum = 0
        rag_sum = 0 
        hybrid_sum = 0
        for model_name in self.evaluator_models.keys():
            model_id = self.evaluator_models[model_name]
            score_finetuning, score_rag, score_hybrid = (self.llm_evaluator_obj).evaluate(model_id,llm_response_finetuning,llm_response_rag, llm_response_hybrid, ground_truth)
            scores['finetuning'][model_name] = score_finetuning 
            scores['rag'][model_name] = score_rag 
            scores['hybrid'][model_name] = score_hybrid 
            finetuning_sum += score_finetuning
            rag_sum += score_rag
            hybrid_sum += score_hybrid
        scores['finetuning']['llm_evaluator_score'] = finetuning_sum/ len(scores['finetuning'].keys())
        scores['rag']['llm_evaluator_score'] = rag_sum/ len(scores['rag'].keys())
        scores['hybrid']['llm_evaluator_score'] = hybrid_sum/ len(scores['hybrid'].keys())

        return scores

    def calculate_scores(self,finetuning_file,rag_file, hybrid_file):
        with open(finetuning_file, 'r', encoding='utf-8') as file:
            finetuning_data = json.load(file)
        with open(rag_file, 'r', encoding='utf-8') as file:
            rag_data = json.load(file)
        with open(hybrid_file, 'r', encoding='utf-8') as file:
            hybrid_data = json.load(file)

        ground_truth = [entry["ground_truth"] for entry in finetuning_data] # ground truth is same for each method

        # Retrieving llm generated texts
        llm_response_finetuning = [entry["llm_response"] for entry in finetuning_data]
        llm_response_rag = [entry["llm_response"] for entry in rag_data]
        llm_response_hybrid = [entry["llm_response"] for entry in hybrid_data]

        # Calculating bert scores
        bert_score_finetuning = self.calculate_bert(ground_truth,llm_response_finetuning)['f1'] #f1, precision and recall is available
        bert_score_rag = self.calculate_bert(ground_truth,llm_response_rag)['f1'] #f1, precision and recall is available
        bert_score_hybrid = self.calculate_bert(ground_truth,llm_response_hybrid)['f1'] #f1, precision and recall is available

        #Calculating LLM Evaluator results
        llm_evaluator_finetuning = []
        llm_evaluator_rag = []
        llm_evaluator_hybrid = []
        for i in range (0,len(ground_truth)):
            llm_evaluator_score  = self.calculate_llm_evaluator(ground_truth[i],llm_response_finetuning[i],llm_response_rag[i], llm_response_hybrid[i])
            
            llm_evaluator_finetuning.append(llm_evaluator_score["finetuning"])
            llm_evaluator_rag.append(llm_evaluator_score["rag"])
            llm_evaluator_hybrid.append(llm_evaluator_score["hybrid"])
        
        return bert_score_finetuning, bert_score_rag, bert_score_hybrid, llm_evaluator_finetuning, llm_evaluator_rag, llm_evaluator_hybrid


    # TODO: Put the scores into a list to make the function call simpler
    def save_scores(self,finetuning_file,rag_file, hybrid_file, bert_score_finetuning, bert_score_rag, bert_score_hybrid, llm_evaluator_finetuning, llm_evaluator_rag, llm_evaluator_hybrid):
        with open(finetuning_file, 'r', encoding='utf-8') as file:
            finetuning_data = json.load(file)
        with open(rag_file, 'r', encoding='utf-8') as file:
            rag_data = json.load(file)
        with open(hybrid_file, 'r', encoding='utf-8') as file:
            hybrid_data = json.load(file)

        for i, entry in enumerate(finetuning_data):
            entry["bert_score"] = bert_score_finetuning[i]
            for key in llm_evaluator_finetuning[i].keys():
                entry[key] = llm_evaluator_finetuning[i][key]

        for i, entry in enumerate(rag_data):
            entry["bert_score"] = bert_score_rag[i]
            for key in llm_evaluator_rag[i].keys():
                entry[key] = llm_evaluator_rag[i][key]

        for i, entry in enumerate(hybrid_data):
            entry["bert_score"] = bert_score_hybrid[i]
            for key in llm_evaluator_hybrid[i].keys():
                entry[key] = llm_evaluator_hybrid[i][key]

        with open(finetuning_file, 'w', encoding='utf-8') as file:
            json.dump(finetuning_data, file, ensure_ascii=False, indent=4)

        with open(rag_file, 'w', encoding='utf-8') as file:
            json.dump(rag_data, file, ensure_ascii=False, indent=4)

        with open(hybrid_file, 'w', encoding='utf-8') as file:
            json.dump(hybrid_data, file, ensure_ascii=False, indent=4)
        
    def calculate_aggregated_scores(self,finetuning_file,rag_file, hybrid_file):
        with open(finetuning_file, 'r', encoding='utf-8') as file:
            finetuning_data = json.load(file)
        with open(rag_file, 'r', encoding='utf-8') as file:
            rag_data = json.load(file)
        with open(hybrid_file, 'r', encoding='utf-8') as file:
            hybrid_data = json.load(file)

        finetuning_bert_score = sum(sample['bert_score'] for sample in finetuning_data) / len(finetuning_data)
        finetuning_llm_eval_score = sum(sample['llm_evaluator_score'] for sample in finetuning_data) / len(finetuning_data)

        rag_bert_score = sum(sample['bert_score'] for sample in rag_data) / len(rag_data)
        rag_llm_eval_score = sum(sample['llm_evaluator_score'] for sample in rag_data) / len(rag_data)


        hybrid_bert_score = sum(sample['bert_score'] for sample in hybrid_data) / len(hybrid_data)
        hybrid_llm_eval_score = sum(sample['llm_evaluator_score'] for sample in hybrid_data) / len(hybrid_data)


        print(f"finetuning_bert_score: {finetuning_bert_score}")
        print(f"finetuning_llm_eval_score: {finetuning_llm_eval_score}")
        
        print("####")

        print(f"rag_bert_score: {rag_bert_score}")
        print(f"rag_llm_eval_score: {rag_llm_eval_score}")

        print("####")

        print(f"hybrid_bert_score: {hybrid_bert_score}")
        print(f"hybrid_llm_eval_score: {hybrid_llm_eval_score}")




