import numpy as np
from collections import Counter
from rouge_score import rouge_scorer
import evaluate
import torch
from bert_score import score as bert_score
from tqdm import tqdm
import pandas as pd
from difflib import SequenceMatcher
import time
import warnings
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Ignore warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
nltk.download('punkt', quiet=True)


import ssl
import certifi
from httpx import Client
ssl._create_default_https_context = ssl._create_unverified_context

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE



def ensure_nltk_downloads():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

# Evaluation metrics functions
def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0

def is_similar(text1, text2, threshold=0.5):
    return SequenceMatcher(None, text1, text2).ratio() > threshold

def compute_precision_at_k(relevant_at_k):
    return safe_divide(sum(relevant_at_k), len(relevant_at_k))

def compute_recall_at_k(relevant_at_k, total_relevant):
    return safe_divide(sum(relevant_at_k), total_relevant)

def compute_mrr(relevant_at_k):
    try:
        first_relevant_rank = next(i for i, r in enumerate(relevant_at_k, 1) if r) + 1
        return 1 / first_relevant_rank
    except StopIteration:
        return 0

def compute_dcg(relevances):
    return sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevances))

def compute_ndcg(relevant_at_k):
    dcg = compute_dcg(relevant_at_k)
    idcg = compute_dcg(sorted(relevant_at_k, reverse=True))
    return safe_divide(dcg, idcg)

def compute_rouge_l(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(reference, candidate)['rougeL'].fmeasure

def compute_bleu(reference, candidate):
    bleu = evaluate.load("bleu")
    return bleu.compute(predictions=[candidate], references=[[reference]])['bleu']
    

def compute_bert_score(reference, candidate):
    _, _, f1 = bert_score([candidate], [reference], lang="en")
    return f1.mean().item()


def compute_exact_match(reference, candidate):
    return int(candidate.strip().lower() == reference.strip().lower())

def compute_f1(reference, candidate):
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    common = Counter(ref_tokens) & Counter(cand_tokens)
    num_common = sum(common.values())
    
    precision = safe_divide(num_common, len(cand_tokens))
    recall = safe_divide(num_common, len(ref_tokens))
    
    return safe_divide(2 * precision * recall, precision + recall)

# RAG system setup functions
def load_vector_db(vector_db_path):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
    return vector_db

def setup_retriever(vector_db, llm):
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_db.as_retriever(search_kwargs={"k": 20})
    )
    return compression_retriever

def setup_llm(api_key, model_name):
    return ChatGroq(
        groq_api_key=api_key,
        model=model_name,
        http_client=Client(verify=ssl_context)
    )

def setup_prompt():
    template = """Answer the question based on the following context:
    {context}
    Question: {question}

    """
    return ChatPromptTemplate.from_template(template)

def load_evaluation_dataset(file_path, sample_size=5):
    df = pd.read_csv(file_path)
    # , encoding='ISO-8859-1'
    # df = df.sample(n=sample_size)
    return df[['Question', 'Answer']].apply(
        lambda row: {"question": row['Question'], "answer": row['Answer']}, axis=1
    ).tolist()


#############################################################################################################################
#############################################################################################################################
#############################################################################################################################


# Main evaluation function
def evaluate_rag_system(evaluation_dataset, retriever, llm, prompt):
    metrics = {
        'precision_at_k': [], 'recall_at_k': [], 'mrr': [], 'ndcg': [],
        'rouge_l': [], 'bleu': [], 'bert_score': [], 'exact_match': [],
        'f1': [], 'response_time': []
    }

    for sample in tqdm(evaluation_dataset):
        question = sample["question"]
        reference_answer = sample["answer"]

        # Retrieve relevant documents
        start_time = time.time()
        relevant_documents = retriever.get_relevant_documents(question)
        # print(relevant_documents)
        retrieval_time = time.time() - start_time

        # Generate answer
        context = "\n".join([doc.page_content for doc in relevant_documents])
        chain = prompt | llm
        start_time = time.time()
        generated_answer = chain.invoke({"context": context, "question": question})
        generation_time = time.time() - start_time

        response_time = retrieval_time + generation_time
        
        # Extract relevant information
        relevant_at_k = [is_similar(doc.page_content, reference_answer) for doc in relevant_documents[:5]]
        
        # Calculate metrics
        metrics['precision_at_k'].append(compute_precision_at_k(relevant_at_k))
        metrics['recall_at_k'].append(compute_recall_at_k(relevant_at_k, len(relevant_documents)))
        metrics['mrr'].append(compute_mrr(relevant_at_k))
        metrics['ndcg'].append(compute_ndcg(relevant_at_k))

        generated_answer_text = generated_answer.content if hasattr(generated_answer, 'content') else str(generated_answer)

        metrics['rouge_l'].append(compute_rouge_l(reference_answer, generated_answer_text))
        metrics['bleu'].append(compute_bleu(reference_answer, generated_answer_text))
        metrics['bert_score'].append(compute_bert_score(reference_answer, generated_answer_text))
        metrics['exact_match'].append(compute_exact_match(reference_answer, generated_answer_text))
        metrics['f1'].append(compute_f1(reference_answer, generated_answer_text))
        metrics['response_time'].append(response_time)

        #create a dataframe and push the results to a csv file
        df = pd.DataFrame(metrics)
        df.to_csv("evaluation_results_inbetween.csv", index=False)

    return metrics

def print_evaluation_results(metrics):
    df = pd.DataFrame(metrics)
    df.to_csv("evaluation_results.csv", index=False)
    for metric, values in metrics.items():
        print(f"Average {metric.replace('_', ' ').title()}: {np.mean(values)}")
    print(f"Number of samples processed: {len(metrics['precision_at_k'])}")



#######################################################################################################################
#######################################################################################################################
#######################################################################################################################


# Main execution
if __name__ == "__main__":
    # Configuration
    vector_db_path = "VECTOR_DB_PATH"
    groq_api_key = "API_KEY"
    model_name = "llama-3.1-70b-versatile"
    evaluation_dataset_path = "QUESTION_CSV_PATH"
        
    # Setup
    vector_db = load_vector_db(vector_db_path)
    llm = setup_llm(groq_api_key, model_name)
    retriever = setup_retriever(vector_db, llm)
    prompt = setup_prompt()
        
    # Load evaluation dataset
    evaluation_dataset = load_evaluation_dataset(evaluation_dataset_path, 5)
        
    # Run evaluation
    metrics = evaluate_rag_system(evaluation_dataset, retriever, llm, prompt)
        
    # Print results
    print_evaluation_results(metrics)