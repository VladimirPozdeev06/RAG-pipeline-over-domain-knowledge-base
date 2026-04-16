from numpy.ma.core import indices
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from TextSplitter import chunk_text
from tqdm import tqdm
import faiss
import pickle
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
def create_embed(path_to_file:str,
                 path_file_to_save_text_chunks:str,path_file_to_save_embed_chunks:str,
                 source:str=None,
                 chunk_size:int=500,overlap:int=50):
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
def search_in_faiss(query:str,top_k:int,faiss_index):
    embed_query=model.encode(query).reshape(1,-1)
    distances,indices=faiss_index.search(embed_query, top_k)
    return distances,indices
def find_relevant_chunks(query:str,top_k:int,faiss_index,all_chunks:list[str],threshold: float = 20.0):
    relevant_chunks=[]
    distances, indices = search_in_faiss(query, top_k, faiss_index)
    for i,item in enumerate(indices[0]):
        if distances[0][i] >= threshold:
            relevant_chunks.append(all_chunks[item])
    return relevant_chunks
def load_chunks():
    faiss_index = faiss.read_index('faiss_index/all_fandom.faiss')
    with open('chunks/all_fandom_chunks.pkl', 'rb') as f:
        all_chunks = pickle.load(f)
    return faiss_index, all_chunks
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
    build_index('chunks/sao_embedding_chunks.npy', 'sao.faiss')'''
    faiss_index=merge_all_faiss_index(['faiss_index/hunter.faiss','faiss_index/naruto.faiss','faiss_index/sao.faiss'],is_save=True,path_to_save='faiss_index/all_fandom.faiss')
    all_chunks=merge_all_chunks(['chunks/hunter_chunks.npy','chunks/naruto_chunks.npy','chunks/sao_chunks.npy'],is_save=True,path_to_save='chunks/all_fandom_chunks.pkl')
    distances,indices=search_in_faiss('Who was the first hokage in Naruto anime?',5,faiss_index)
    print(distances)
    print(indices)
    for item in indices[0]:
        print(all_chunks[item])



