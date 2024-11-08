
"""
BLEURT SCORE
Info -> https://github.com/google-research/bleurt
Installation: 
- git clone https://github.com/google-research/bleurt.git
- cd bleurt
- pip install .

BLEURT-20 Installation
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip
python -m bleurt.score_files \
  -candidate_file=bleurt/test_data/candidates \
  -reference_file=bleurt/test_data/references \
  -bleurt_checkpoint=BLEURT-20



"""
from bleurt import score
import tensorflow as tf
import json
from evaluate import load
import numpy as np


def calculate_bert(ground_truth,llm_generated):
    bertscore = load("bertscore")
    scores = bertscore.compute(predictions=llm_generated, references=ground_truth, lang="en")
    return scores

def calculate_bleurt(ground_truth, llm_generated):
    test_checkpoint = 'bleurt/bleurt/test_checkpoint' # this is inaccurate, use BLEURT-20
    checkpoint = 'bleurt/BLEURT-20'

    scorer = score.BleurtScorer(checkpoint)
    scores = scorer.score(references=ground_truth, candidates=llm_generated)
    #assert isinstance(scores, list) and len(scores) == 1
    return scores

def calculate_scores(data_file):
    with open(data_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

        llm_response = [entry["llm_response"] for entry in data]
        ground_truth = [entry["ground_truth"] for entry in data]

        bleurt_score = calculate_bleurt(ground_truth,llm_response)
        bert_score = calculate_bert(ground_truth,llm_response)['f1'] #f1, precision and recall is available

    for i, entry in enumerate(data):
        entry["bleurt_score"] = bleurt_score[i]
        entry["bert_score"] = bert_score[i]

    with open(data_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"Avg Bleurt Score: {np.mean(bleurt_score)}")
    print(f"Avg Bert Score: {np.mean(bert_score)}")


# TODO: Combine all scores in this function
def evaluate(ground_truth, llm_response):
    print("IMPLEMENT ME! Evaluation function")
    return 0.8

if __name__ == "__main__":

    print("# FINETUNING")
    finetuning_results = 'data/output/instruction_finetuning_results.json'
    calculate_scores(finetuning_results)
    print("####")

    print("# RAG")
    rag_results = 'data/output/rag_results.json'
    calculate_scores(rag_results)
    print("####")

    print("# HYBRID")
    rag_results = 'data/output/hybrid_results.json'
    calculate_scores(rag_results)

