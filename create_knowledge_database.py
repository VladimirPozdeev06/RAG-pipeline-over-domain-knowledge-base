from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import numpy as np
from TextSplitter import chunk_text
from tqdm import tqdm
import faiss
from rank_bm25 import BM25Okapi
import pickle
import time
from typing import Literal
from collections import defaultdict
from prepare_data import detect_english_text, lemmatize_en, lemmatize_ru
MODELS = {
    'bge-m3': {'model': SentenceTransformer("BAAI/bge-m3")},
    'multi-e5': {'model': SentenceTransformer("intfloat/multilingual-e5-base")}
}
reranker = CrossEncoder('BAAI/bge-reranker-base')
def create_embed(path_to_file:str,
                 path_file_to_save_text_chunks:str,path_file_to_save_embed_chunks:str,
                 source:str=None,
                 chunk_size:int=500,overlap:int=50,
                 retriever_model:Literal['bge-m3','multi-e5']='bge-m3',
                 ):
    model=MODELS[retriever_model]['model']
    data = []
    chunks=[]
    with open(path_to_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line:
                data.append(json.loads(line))


    for page in tqdm(data):
        text='Title: ' + page['title'] + '\n'+page['clean_text']
        for chunk in chunk_text(text,chunk_size,overlap):
            chunks.append({'title':page['title'], 'text':chunk,'source':source})


    if retriever_model=='multi-e5':
        embed_chunks = model.encode(['passage: '+c['text'] for c in chunks], show_progress_bar=True)
    else:
        embed_chunks=model.encode([c['text'] for c in chunks], show_progress_bar=True)
    embed_chunks=np.array(embed_chunks)
    np.save(path_file_to_save_text_chunks,chunks)
    np.save(path_file_to_save_embed_chunks, embed_chunks)

def build_index(path_to_embedded_chunks:str,path_to_save_index:str):
    embed_chunks=np.load(path_to_embedded_chunks)
    index=faiss.IndexFlatIP(embed_chunks.shape[1])
    index.add(embed_chunks)
    faiss.write_index(index, path_to_save_index)

def merge_all_faiss_index(list_path_faiss_dataset:list[str],is_save:bool=False,path_to_save:str=None):
    for i,path in enumerate(list_path_faiss_dataset):
        if i==0:
            merged_index=faiss.read_index(path)
        else:
            index_to_merge=faiss.read_index(path)
            merged_index.merge_from(index_to_merge)
    if is_save:
        faiss.write_index(merged_index, path_to_save)

    return merged_index
def merge_all_chunks(list_path_to_numpy_arr_with_chunks,is_save:bool=False,path_to_save:str=None):
    all_chunks = []
    for path in list_path_to_numpy_arr_with_chunks:
        all_chunks.extend(np.load(path, allow_pickle=True))
    if is_save:
        with open(path_to_save, 'wb') as f:
            pickle.dump(all_chunks, f)
    return all_chunks
def search_in_faiss(query,top_k:int,faiss_index,all_chunks,threshold: float| None = None,show_time:bool=False,return_time:bool=False,retriever_model:Literal['bge-m3','multi-e5']='bge-m3'):
    model = MODELS[retriever_model]['model']
    if retriever_model=='multi-e5':
        query='query: ' + query
    embed_query = model.encode(query).reshape(1, -1)
    start_time = time.perf_counter()
    distances,indices=faiss_index.search(embed_query, top_k)
    end_time = time.perf_counter()
    if show_time:
        print(end_time-start_time)
    relevant_chunks = []
    for i, item in enumerate(indices[0]):
        if threshold is None or distances[0][i] >= threshold:
            relevant_chunks.append(all_chunks[item])
    if return_time:
        return relevant_chunks,end_time-start_time
    return relevant_chunks
def search_bm_25(query:str,top_k:int,all_chunks=None,bm25_index=None,tokenized_chunks=None,show_time:bool=False,return_time:bool=False):
    if detect_english_text(query, min_confidence=0.8):
        split_query = lemmatize_en(query)
    else:
        split_query = lemmatize_ru(query)
    #tokenized_chunks=[c['text'].lower().split() for c in all_chunks]
    start_time = time.perf_counter()

    relevant_chunks=bm25_index.get_top_n(split_query,all_chunks,top_k)
    end_time = time.perf_counter()
    if show_time:
        print(end_time-start_time)
    if return_time:
        return relevant_chunks,end_time-start_time
    return relevant_chunks
def rerank(query:str,relevant_chunks:list,top_k:int):
    pairs=[(query,c['text']) for c in relevant_chunks]
    scorer=reranker.predict(pairs)
    ranked=sorted(zip(scorer,relevant_chunks),key = lambda x: x[0],reverse=True)
    return [chunk for score,chunk in ranked[:top_k]]


def find_relevant_chunks(query: str,
                         top_k: int,

                         retriever_type: Literal['faiss', 'bm25'],
                         faiss_index, all_chunks: list[str], tokenized_chunks,
                         retriever_model: Literal['bge-m3', 'multi-e5'] = 'bge-m3',
                         is_hybrid: bool = False,
                         k_rrf_hybrid: int = 60,
                         threshold: float | None = None,
                         use_reranker: bool = False,
                         top_k_for_reranker: int = 5):
    if is_hybrid:
        relevant_chunks_faiss = search_in_faiss(query, top_k, faiss_index, all_chunks, threshold,
                                                retriever_model=retriever_model)
        relevant_chunks_faiss_ranks = [(1 / (k_rrf_hybrid + rank), chunk) for rank, chunk in
                                       enumerate(relevant_chunks_faiss)]

        relevant_chunks_bm25 = search_bm_25(query, top_k, all_chunks, tokenized_chunks)
        relevant_chunks_bm25_ranks = [(1 / (k_rrf_hybrid + rank), chunk) for rank, chunk in
                                      enumerate(relevant_chunks_bm25)]

        scores = defaultdict(float)
        id_to_chunk = {}

        for score, chunk in relevant_chunks_faiss_ranks:
            scores[id(chunk)] += score
            id_to_chunk[id(chunk)] = chunk

        for score, chunk in relevant_chunks_bm25_ranks:
            scores[id(chunk)] += score
            id_to_chunk[id(chunk)] = chunk

        relevant = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        relevant_chunks = [id_to_chunk[cid] for cid, score in relevant[:top_k]]


    else:
        if retriever_type == 'faiss':
            relevant_chunks = search_in_faiss(query, top_k, faiss_index, all_chunks, threshold,
                                              retriever_model=retriever_model)
        else:
            relevant_chunks = search_bm_25(query, top_k, all_chunks, tokenized_chunks)
    if use_reranker:
        relevant_chunks = rerank(query, relevant_chunks, top_k_for_reranker)
    return relevant_chunks

if __name__ == '__main__':
    '''create_embed('parsed_pages/hunter_parsed_pages.jsonl',
                 'hunter_chunks.npy',
                 'hunter_embedding_chunks.npy',source='hunter')
    arr1=np.load('hunter_embedding_chunks.npy')
    arr2=np.load('hunter_chunks.npy',allow_pickle=True)
    print(len(arr1),len(arr2))

    create_embed('parsed_pages/naruto_parsed_pages.jsonl',
                 'naruto_chunks.npy',
                 'naruto_embedding_chunks.npy',source='naruto')

    create_embed('parsed_pages/sao_parsed_pages.jsonl',
                 'sao_chunks.npy',
                 'sao_embedding_chunks.npy',source='sao')

    build_index('chunks/hunter_embedding_chunks.npy', 'hunter.faiss')
    build_index('chunks/naruto_embedding_chunks.npy', 'naruto.faiss')
    build_index('chunks/sao_embedding_chunks.npy', 'sao.faiss')
    faiss_index=merge_all_faiss_index(['faiss_index/hunter.faiss','faiss_index/naruto.faiss','faiss_index/sao.faiss'],is_save=True,path_to_save='faiss_index/all_fandom.faiss')
    all_chunks=merge_all_chunks(['chunks/hunter_chunks.npy','chunks/naruto_chunks.npy','chunks/sao_chunks.npy'],is_save=True,path_to_save='chunks/all_fandom_chunks.pkl')
    distances,indices=search_in_faiss('Who was the first hokage in Naruto anime?',5,faiss_index)
    print(distances)
    print(indices)
    for item in indices[0]:
        print(all_chunks[item])'''



