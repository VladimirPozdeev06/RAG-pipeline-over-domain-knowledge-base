import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness,answer_correctness,answer_relevancy
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
def recall_k(list_of_relevant_chunks:list,list_of_chunks:list,top_k:int)->float:
    k_chunks=list_of_chunks[:top_k]
    number_relevant_chunks_in_top_k=len(set(k_chunks) &set(list_of_relevant_chunks))
    number_relevant_chunks=len(list_of_relevant_chunks)
    return round(number_relevant_chunks_in_top_k/ number_relevant_chunks,3)
def precision_k(list_of_relevant_chunks:list,list_of_chunks:list,top_k:int)->float:
    k_chunks=list_of_chunks[:top_k]
    number_relevant_chunks_in_top_k = len(set(k_chunks) & set(list_of_relevant_chunks))
    return round(number_relevant_chunks_in_top_k/top_k,3)
def hit_k(list_of_relevant_chunks:list,list_of_chunks:list,top_k:int)->int:
    k_chunks=list_of_chunks[:top_k]
    number_relevant_chunks_in_top_k = len(set(k_chunks) & set(list_of_relevant_chunks))
    hit=1 if number_relevant_chunks_in_top_k>0 else 0
    return hit

def nDCG_k(list_of_relevant_chunks:list,list_of_chunks:list,top_k:int)->float:
    k_chunks=list_of_chunks[:top_k]
    DCG,IDCG=0,0
    ideal_length = min(top_k, len(list_of_relevant_chunks))
    IDCG = sum(1 / np.log2(rank + 1) for rank in range(1, ideal_length + 1))
    for rank,chunk in enumerate(k_chunks,start=1):

        if chunk in list_of_relevant_chunks:
            DCG+=(1/np.log2(rank+1))
    return round(DCG/IDCG,3)

def MRR(list_of_relevant_chunks:list,list_of_chunks:list)->float:
    for rank,chunk in enumerate(list_of_chunks):
        if chunk in list_of_relevant_chunks:
            return 1/(rank+1)
    return 0.0

def context_precision(list_of_relevant_chunks:list,list_of_chunks:list):
    number_relevant_chunks_in_top_k=len(set(list_of_relevant_chunks) &set(list_of_chunks))
    sum_precision_k=0
    relevant_c=0
    for rank,chunk in enumerate(list_of_chunks,start=1):
        if chunk in list_of_relevant_chunks:
            relevant_c+=1
            sum_precision_k += relevant_c/rank
            #sum_precision_k+=precision_k(list_of_relevant_chunks,list_of_chunks,top_k=rank)
    return round(sum_precision_k/number_relevant_chunks_in_top_k,3)

def compute_generation_metrics(queries:list[str],answers:list[str],chunks_from_model:list[list[str]],llm):
    data_samples={
        'question':queries,
        'answer':answers,
        'context':chunks_from_model
    }
    data=Dataset.from_dict(data_samples)
    score=evaluate(data,metrics=[faithfulness,answer_correctness,answer_relevancy],llm=llm,show_progress=True)
    return score.to_pandas()

def get_groq_llm_for_evaluate_using_ragas(model_name:str='llama-3.3-70b-versatile'):
    groq_llm=ChatGroq(model=model_name,
                      api_key=os.getenv('GROQ_API_KEY'),
                      temperature=0)
    return LangchainLLMWrapper(groq_llm)
