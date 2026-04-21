import json
import ast
from typing import Literal
import pandas as pd
from implement_LLM import oracle_retriever
from metrics import compute_all_metrics
def parse_chunks(val):
    try:
        return json.loads(val)
    except:
        return ast.literal_eval(val)
def complete_eval_pipline(
    path_to_eval_set:str='eval_set_v3_clean_sampled.jsonl',
    is_oracle:bool=False,
    is_print_info:bool=False,
    is_generate_answers:bool=False,
    path_to_data_with_answers:str=None,
    top_k: int = 5,
    retriever_type: Literal["faiss", "bm25"] = "faiss",
    faiss_index=None,
    all_chunks=None,
    name_generation_model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
    max_tokens: int = 256,
    threshold: float = 20.0,
    show_time: bool = False,
    return_time: bool = False,
    is_oracle_retriever: bool = False,
    relevant_chunk_ids=None,
    use_few_shot: bool = True,
    is_generation_metrics: bool = False,
    is_simple_generation_metrics: bool = False,
    name_evaluation_model: str = 'llama-3.3-70b-versatile',
    queries_column: str = 'question',
    answers_column: str = 'llm_answer',
    chunks_column: str ='relevant_chunks',
    ground_truth_column: str = 'ground_truth_text',
    relevant_chunks_column: str = None,
    queries: list[str] = None,
    answers: list[str] = None,
    chunks_from_model: list[list[str]] = None,
    ground_truth_text: list[str] = None,

    # retrieval metrics
    is_retrieval_metrics: bool = False,

    top_k_recall: list[int] = None,
    top_k_precision: list[int] = None,
    top_k_hit: list[int] = None,
    top_k_nDCG: list[int] = None,
    is_context_precision: bool = False,

    is_time_metrics: bool = False,
    list_columns_time: list[str] =None,
    is_abstention_metrics :bool=False,

    save_metrics_data:bool=False,
    path_to_save_metrics_data:str=None,
     ):


    if is_oracle:
        if is_generate_answers:
            data_oracle = oracle_retriever(path_to_eval_set, all_chunks=None,
                                       name_model=name_generation_model)
        else:
            data_oracle = pd.read_csv(path_to_data_with_answers)
        if is_print_info:
            print(data_oracle.info())
        data_oracle['relevant_chunks'] = data_oracle['relevant_chunks'].apply(parse_chunks)
        if list_columns_time is  None:
            list_columns_time=['generation_time', 'e2e_latency']
        result = compute_all_metrics(is_simple_generation_metrics=is_simple_generation_metrics,
                                     is_generation_metrics=is_generation_metrics,
                                     data_samples=data_oracle,
                                     model_name=name_evaluation_model,
                                     queries_column=queries_column,
                                     answers_column=answers_column,
                                     chunks_column=chunks_column,
                                     ground_truth_column=ground_truth_column,
                                     is_time_metrics=is_time_metrics,
                                     list_columns_time=list_columns_time,
                                     is_abstention_metrics=is_abstention_metrics,
                                     is_retrieval_metrics=False
                                     )
        if is_generation_metrics:
            generation_df = result['generation'][['faithfulness', 'answer_correctness', 'answer_relevancy']]
            data_oracle_with_metrics = pd.concat([data_oracle.reset_index(drop=True), generation_df.reset_index(drop=True)],
                                                 axis=1)
            if path_to_save_metrics_data:
                data_oracle_with_metrics.to_csv(path_to_save_metrics_data, index=False)
            return data_oracle_with_metrics,result
        else:
            return data_oracle,result
    else:
        return 0