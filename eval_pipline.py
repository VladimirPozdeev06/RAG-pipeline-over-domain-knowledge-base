import json
import ast
from typing import Literal
import pandas as pd
from implement_LLM import oracle_retriever, generate_response
from metrics import compute_all_metrics
def parse_chunks(val):
    try:
        return json.loads(val)
    except:
        return ast.literal_eval(val)
def complete_eval_pipline(
    path_to_eval_set:str='eval_set_v3_clean_sampled.jsonl',
    is_oracle:bool=False,
    is_alone_retriever:bool=False,
    is_print_info:bool=False,
    is_generate_answers:bool=False,
    path_to_data_with_answers_oracle:str=None,
    top_k_chunks: int = 5,
    retriever_type: Literal["faiss", "bm25"] = "faiss",
    retriever_model:Literal['bge-m3','multi-e5']='bge-m3',
    faiss_index=None,
    all_chunks=None,
    name_generation_model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
    max_tokens: int = 256,
    threshold: float = 20.0,
    show_time: bool = False,
    return_time: bool = True,


    use_few_shot: bool = True,
    is_generation_metrics: bool = False,
    is_simple_generation_metrics: bool = False,
    name_evaluation_model: str = 'llama-3.3-70b-versatile',
    queries_column: str = 'question',
    answers_column: str = 'llm_answer',
    chunks_column: str ='relevant_chunks',
    ground_truth_column: str = 'ground_truth_text',
    relevant_chunks_column: str = 'relevant_chunks',


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
    path_to_data_with_answers_retriever_alone:str=None
     ):
    if not is_oracle and not is_alone_retriever:
        raise ValueError('Нужно указать is_oracle=True или is_alone_retriever=True')
    if is_time_metrics and  list_columns_time is None:
        list_columns_time = ['generation_time', 'e2e_latency']
    if is_oracle:
        if is_generate_answers:
            data = oracle_retriever(path_to_eval_set, all_chunks=None,
                                       name_model=name_generation_model)
        else:
            data = pd.read_csv(path_to_data_with_answers_oracle)
        if is_print_info:
            print(data.info())
        data[relevant_chunks_column] = data[relevant_chunks_column].apply(parse_chunks)

        metrics_result = compute_all_metrics(is_simple_generation_metrics=is_simple_generation_metrics,
                                     is_generation_metrics=is_generation_metrics,
                                     data_samples=data,
                                     model_name=name_evaluation_model,
                                     queries_column=queries_column,
                                     answers_column=answers_column,
                                     chunks_column=chunks_column,
                                     ground_truth_column=ground_truth_column,
                                     is_time_metrics=is_time_metrics,
                                     list_columns_time=list_columns_time,
                                     is_abstention_metrics=is_abstention_metrics,
                                     is_retrieval_metrics=False,

                                     )

    if is_alone_retriever:
        if is_generate_answers:
            data=pd.read_json(path_to_eval_set,lines=True)
            generation_results=data.progress_apply(
                lambda x:generate_response(
                        query=x['question'],
                        top_k=top_k_chunks,
                        all_chunks=all_chunks,
                        faiss_index=faiss_index,
                        retriever_type=retriever_type,
                        name_model=name_generation_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        return_time=return_time,
                        is_oracle_retriever=False,
                        relevant_chunk_ids=x["relevant_chunk_ids"],
                        use_few_shot=use_few_shot,
                        threshold = threshold,
                        show_time=show_time,
                        retriever_model=retriever_model

                        ),axis=1)
            if return_time:
                data["llm_answer"], data["generation_time"], data["e2e_latency"],data[relevant_chunks_column]=zip(*generation_results)
            else:
                data["llm_answer"],data[relevant_chunks_column]=zip(*generation_results)
        else:
            data=pd.read_csv(path_to_data_with_answers_retriever_alone)
        if is_print_info:
            print(data.info())
        metrics_result = compute_all_metrics(is_simple_generation_metrics=is_simple_generation_metrics,
                                     is_generation_metrics=is_generation_metrics,
                                     data_samples=data,
                                     model_name=name_evaluation_model,
                                     queries_column=queries_column,
                                     answers_column=answers_column,
                                     chunks_column=chunks_column,
                                     ground_truth_column=ground_truth_column,
                                     is_time_metrics=is_time_metrics,
                                     list_columns_time=list_columns_time,
                                     is_abstention_metrics=is_abstention_metrics,
                                     is_retrieval_metrics=is_retrieval_metrics,
                                     top_k_recall = top_k_recall,
                                     top_k_precision=top_k_precision,
                                     top_k_hit = top_k_hit,
                                     top_k_nDCG=top_k_nDCG,
                                     is_context_precision=is_context_precision,

                                     )

    if is_generation_metrics:
        generation_df = metrics_result['generation'][['faithfulness', 'answer_correctness', 'answer_relevancy']]
        data_with_metrics = pd.concat([data.reset_index(drop=True), generation_df.reset_index(drop=True)],
                                             axis=1)
        if path_to_save_metrics_data:
            data_with_metrics.to_csv(path_to_save_metrics_data, index=False)
        return data_with_metrics,  metrics_result
    else:
        return data,  metrics_result
