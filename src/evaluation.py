
from bleurt import score
import tensorflow as tf
import json
from evaluate import load
import numpy as np


def calculate_bert(ground_truth,llm_generated):
    bertscore = load("bertscore")
    scores = bertscore.compute(predictions=llm_generated, references=ground_truth, lang="en")
    return scores


def calculate_scores(data_file):
    with open(data_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

        llm_response = [entry["llm_response"] for entry in data]
        ground_truth = [entry["ground_truth"] for entry in data]

        bert_score = calculate_bert(ground_truth,llm_response)['f1'] #f1, precision and recall is available

    for i, entry in enumerate(data):
        entry["bert_score"] = bert_score[i]

    with open(data_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

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

