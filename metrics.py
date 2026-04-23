import numpy as np
import pandas as pd
import re
import string
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness,answer_correctness,answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import RunConfig
from bert_score import score as bert_score
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

def recall_k(list_of_relevant_chunks:list,list_of_chunks:list,top_k:int)->float:
    k_chunks=list_of_chunks[:top_k]
    number_relevant_chunks_in_top_k=len(set(k_chunks) &set(list_of_relevant_chunks))
    number_relevant_chunks=len(list_of_relevant_chunks)
    if number_relevant_chunks == 0:
        return 0.0
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
    if IDCG == 0:
        return 0.0
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
    if number_relevant_chunks_in_top_k == 0:
        return 0.0
    sum_precision_k=0
    relevant_c=0
    for rank,chunk in enumerate(list_of_chunks,start=1):
        if chunk in list_of_relevant_chunks:
            relevant_c+=1
            sum_precision_k += relevant_c/rank
    return round(sum_precision_k/number_relevant_chunks_in_top_k,3)

def generation_metrics(llm,
                               data_samples:pd.DataFrame=None,
                               queries_column:str=None,
                               answers_column:str=None,
                               chunks_column:str=None,
                               ground_truth_column:str=None,
                               queries:list[str]=None,
                               answers:list[str]=None,
                               chunks_from_model:list[list[str]]=None,
                               ground_truth_text:list[str]=None):
    if data_samples is not None :
        if queries_column is None or answers_column is None or chunks_column is None or ground_truth_column is None:
            raise ValueError('Columns for data should be provided.')
        data_samples=data_samples.rename(columns={queries_column:'user_input',
                                                  answers_column:'response',
                                                  chunks_column:'retrieved_contexts',
                                                  ground_truth_column:'reference'})
        data = Dataset.from_pandas(data_samples)


    else:
        data_samples={
            'user_input':queries,
            'response':answers,
            'retrieved_contexts':chunks_from_model,
            'reference':ground_truth_text
        }
        data=Dataset.from_dict(data_samples)
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    faithfulness.llm = llm
    answer_correctness.llm = llm
    answer_relevancy.llm = llm
    answer_relevancy.embeddings = embeddings
    answer_relevancy.strictness = 1
    score=evaluate(data,
                   metrics=[faithfulness,answer_correctness,answer_relevancy],
                   llm=llm,
                   show_progress=True,
                   embeddings=embeddings,
                   run_config=RunConfig(max_workers=1, timeout=300))
    return score.to_pandas()

def get_groq_llm_for_evaluate_using_ragas(model_name:str='llama-3.3-70b-versatile'):
    groq_llm=ChatGroq(model=model_name,
                      api_key=os.getenv('GROQ_API_KEY'),
                      temperature=0)
    return LangchainLLMWrapper(groq_llm)

def compute_time_metrics(data:pd.DataFrame,list_columns_time:list[str]):
    for col in list_columns_time:
        print(f'{col} mean: {data[col].mean()}')
        print(f'{col} median: {data[col].median()}')
        print(f'{col} p95 : {data[col].quantile(0.95)}')

def normalize(s):
    s = str(s).lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s
def exact_match(answer: str, gt: str) -> int:

    return int(answer == gt)
def compute_simple_generation_metrics(data:pd.DataFrame,answer_col:str,ground_truth_col:str):
    answer_text = data[answer_col].fillna('<empty>').astype(str).apply(normalize).tolist()
    ground_truth_text = data[ground_truth_col].fillna('<empty>').astype(str).apply(normalize).tolist()

    answer_text = [a if a.strip() else '<empty>' for a in answer_text]
    ground_truth_text = [g if g.strip() else '<empty>' for g in ground_truth_text]
    data['exact_match']=[exact_match(answer,gt) for answer,gt in zip(answer_text,ground_truth_text)]

    P,R,F1=bert_score(answer_text,ground_truth_text,model_type='xlm-roberta-large',lang='other',verbose=True)

    data['bertscore_precision'] = P.numpy()
    data['bertscore_recall'] = R.numpy()
    data['bertscore_f1'] = F1.numpy()

    print(f"Exact Match: {data['exact_match'].mean():.3f}")
    print(f"BERTScore F1: {data['bertscore_f1'].mean():.3f}")
    print(f"BERTScore Precision: {data['bertscore_precision'].mean():.3f}")
    print(f"BERTScore Recall: {data['bertscore_recall'].mean():.3f}")

def is_abstain(s):
    s = str(s).lower()
    return any(k in s for k in ['not in', 'нет данных', 'cannot', 'не могу', 'no data'])
def compute_all_metrics(       # generation_metrics
                               is_generation_metrics:bool=False,
                               is_simple_generation_metrics:bool=False,
                               model_name:str='llama-3.3-70b-versatile',
                               data_samples:pd.DataFrame=None,
                               queries_column: str = None,
                               answers_column: str = None,
                               chunks_column: str = None,
                               ground_truth_column: str = None,
                               relevant_chunks_column: str = None,
                               answer_type_column:str='answer_type',
                               queries:list[str]=None,
                               answers:list[str]=None,
                               chunks_from_model:list[list[str]]=None,
                               ground_truth_text:list[str]=None,

                               #retrieval metrics
                               is_retrieval_metrics: bool = False,

                              
                               top_k_recall:list[int]=None,
                               top_k_precision:list[int]=None,
                               top_k_hit:list[int]=None,
                               top_k_nDCG:list[int]=None,
                               is_context_precision:bool=False,

                               is_time_metrics:bool=False,
                               list_columns_time:list[str]=None,

                               is_abstention_metrics:bool=False

                               ):


    results = {}
    if is_generation_metrics:
        llm=get_groq_llm_for_evaluate_using_ragas(model_name=model_name)
        score=generation_metrics(llm,
                               data_samples=data_samples,
                               queries=queries,
                               answers=answers,
                               chunks_from_model=chunks_from_model,
                               ground_truth_text=ground_truth_text,
                               queries_column = queries_column,
                               answers_column =answers_column,
                               chunks_column = chunks_column,
                               ground_truth_column=ground_truth_column
        )
        print(score.mean(numeric_only=True))
        results['generation'] = score

    if is_simple_generation_metrics:
        compute_simple_generation_metrics(data_samples,answers_column,ground_truth_column)

    if is_time_metrics:
        if data_samples is None or list_columns_time is None:
            raise ValueError('Both data_samples and list_columns_time should be provided.')
        compute_time_metrics(data_samples,list_columns_time)
        results['time'] = {col: {'mean': data_samples[col].mean(),
                                 'median': data_samples[col].median(),
                                 'p95': data_samples[col].quantile(0.95)}
                           for col in list_columns_time}

    if is_retrieval_metrics:
        if data_samples is None:
            raise ValueError(" data_samples  must be provided")
        retrieval_results = {}

        if top_k_recall is not None:
            for k in top_k_recall:
                mean_recall=data_samples.apply(
                    lambda x: recall_k(x[relevant_chunks_column], x[chunks_column],k),axis=1
                ).mean()
                print(f'top_{k}_recall: {mean_recall}')
                retrieval_results[f'top_{k}_recall'] = mean_recall

        if top_k_precision is not None:
            for k in top_k_precision:
                mean_precision = data_samples.apply(
                    lambda x: precision_k(x[relevant_chunks_column], x[chunks_column], k), axis=1
                ).mean()
                print(f'top_{k}_precision: {mean_precision}')
                retrieval_results[f'top_{k}_precision'] = mean_precision

        if top_k_hit is not None:
            for k in top_k_hit:
                mean_hit = data_samples.apply(
                    lambda x: hit_k(x[relevant_chunks_column], x[chunks_column], k), axis=1
                ).mean()
                print(f'top_{k}_hit: {mean_hit}')
                retrieval_results[f'top_{k}_hit'] = mean_hit

        if top_k_nDCG is not None:
            for k in top_k_nDCG:
                mean_nDCG = data_samples.apply(
                    lambda x: nDCG_k(x[relevant_chunks_column], x[chunks_column], k), axis=1
                ).mean()
                print(f'top_{k}_nDCG: {mean_nDCG}')
                retrieval_results[f'top_{k}_nDCG'] = mean_nDCG

        if is_context_precision:
            mean_context_precision=data_samples.apply(
                lambda x: context_precision(x[relevant_chunks_column],x[chunks_column]),axis=1
            ).mean()
            print(f'context_precision: {mean_context_precision}')
            retrieval_results['context_precision'] = mean_context_precision

        results['retrieval'] = retrieval_results

    if is_abstention_metrics:
        abst_mask = data_samples[answer_type_column] == 'abstention'
        abst_acc = data_samples.loc[abst_mask, answers_column].apply(is_abstain).mean()
        print(f'Точность правильных отказов: {abst_acc:.1%}')

        non_abst = data_samples[data_samples[answer_type_column] != 'abstention']
        false_abstain = non_abst[answers_column].apply(is_abstain).mean()
        print(f'Точность ложных отказов: {false_abstain:.1%}')

        results['abstention'] = {
            'abstain_accuracy': abst_acc,
            'false_abstain_rate': false_abstain
        }

    return results if results else None

def compute_agg_metrics(
        data:pd.DataFrame,
        #generation metrics
        is_generation_metrics:bool=False,
        aggregation_generation_columns:list=None,
        generation_metrics_columns:list=None,

        is_simple_generation_metrics: bool = False,
        simple_aggregation_generation_columns: list = None,
        simple_generation_metrics_columns: list = None,
        #time metrics
        is_time_metrics:bool=False,
        aggregation_time_columns:list=None,
        time_metrics_columns:list=None,

        #retriever metrics
        is_retriever_metrics:bool=False,
        aggregation_retriever_columns:list=None,
        retriever_metrics_columns:list=None,
        ):
    if is_generation_metrics:
        if generation_metrics_columns is None:
            generation_metrics_columns=['faithfulness','answer_correctness','answer_relevancy']
        for agg_column in aggregation_generation_columns:
            agg_result=data.groupby(agg_column)[generation_metrics_columns].mean()
            print(agg_result)

    if is_simple_generation_metrics:
        if simple_generation_metrics_columns is None:
            simple_generation_metrics_columns=['bertscore_precision','bertscore_recall','bertscore_f1','exact_match']
        for agg_column in simple_aggregation_generation_columns:
            agg_result=data.groupby(agg_column)[simple_generation_metrics_columns].mean()
            print(agg_result)

    if is_time_metrics:
        if time_metrics_columns is None:
            time_metrics_columns=['e2e_latency','generation_time']
        for agg_column in aggregation_time_columns:
            agg_result=data.groupby(agg_column)[time_metrics_columns].mean()
            print(agg_result)

    if is_retriever_metrics:
        if retriever_metrics_columns is None:
            retriever_metrics_columns=['context_precision']
        for agg_column in aggregation_retriever_columns:
            agg_result=data.groupby(agg_column)[retriever_metrics_columns].mean()
            print(agg_result)

