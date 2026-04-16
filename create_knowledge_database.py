from sentence_transformers import SentenceTransformer
import json
import numpy as np
from TextSplitter import chunk_text
from tqdm import tqdm
import faiss
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
                 'sao_embedding_chunks.npy',source='sao')'''

    build_index('chunks/hunter_embedding_chunks.npy', 'hunter.faiss')
    build_index('chunks/naruto_embedding_chunks.npy', 'naruto.faiss')
    build_index('chunks/sao_embedding_chunks.npy', 'sao.faiss')




