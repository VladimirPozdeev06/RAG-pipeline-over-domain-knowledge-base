import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from typing import Literal
from groq import Groq
import faiss
import pickle
import time
from create_knowledge_database import find_relevant_chunks
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)
def generate_response(query:str,
                      top_k:int=5,
                      retriever_type:Literal['faiss','bm25']='faiss',
                      faiss_index=None,all_chunks=None,
                      name_model:str='llama-3.3-70b-versatile',
                      temperature:float=0.3,
                      max_tokens:int=512,
                      threshold:float=20.0,
                      show_time:bool=False,
                      return_time:bool=False,
                      is_oracle_retriever:bool=False,
                      relevant_chunk_ids:int|list|None=None
                      ):
    if faiss_index is None:
        faiss_index = faiss.read_index('faiss_index/all_fandom.faiss')
    if all_chunks is None:
        with open('chunks/all_fandom_chunks.pkl', 'rb') as f:
            all_chunks=pickle.load(f)
    relevant_chunks=[]
    start_e2e_latency_time=time.perf_counter()
    if not is_oracle_retriever:
        relevant_chunks=[rc['text'] for rc in find_relevant_chunks( query, top_k, retriever_type,faiss_index,all_chunks,threshold)]
    else:
        if type(relevant_chunk_ids) is  int:
            relevant_chunks=[all_chunks[relevant_chunk_ids]['text']]
        elif type(relevant_chunk_ids) is  list:
            relevant_chunks=[all_chunks[ids]['text'] for ids in relevant_chunk_ids]
    if len(relevant_chunks)==0:
        return "I can only answer questions about Hunter x Hunter, Naruto, and Sword Art Online."
    start_generate_time = time.perf_counter()
    response=client.chat.completions.create(
       model=name_model,
       messages=[
           {
               'role':'system',
               'content': 'You are expert in anime topic. Answer on question using provided content. You should obligatory use it and notify if he can not find an answer or information can be incomplete. Moreover if you find opposite information you should also say about it. Do not lye and imagine. Response only on query question.Do not mention provided knowledge base.Be concise. Answer in 3-5 sentences.'
           },
           {
               'role':'user',
               'content': f'I have a question: {query}. Give me an answer base on next knowledge base: {relevant_chunks}'
           }
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    time.sleep(2)
    text=response.choices[0].message.content
    end_generate_time = time.perf_counter()
    end_e2e_latency_time = time.perf_counter()
    generation_time=end_generate_time-start_generate_time
    e2e_latency=end_e2e_latency_time-start_e2e_latency_time

    if show_time:
        print(f'generate_time: {generation_time} seconds')
        print(f'e2e_latency: {e2e_latency} seconds')
    if is_oracle_retriever:
        return text,generation_time,e2e_latency,relevant_chunks
    if return_time:
        return text,generation_time,e2e_latency
    return text
def load_chunks(path_faiss_embed_chunks:str='faiss_index/all_fandom.faiss',
                path_text_chunks:str='chunks/all_fandom_chunks.pkl'):
    faiss_index = faiss.read_index((path_faiss_embed_chunks))
    with open(path_text_chunks, 'rb') as f:
        all_chunks = pickle.load(f)
    return faiss_index, all_chunks

def oracle_retriever(path_data_to_eval:str,all_chunks:list=None,
                     name_model:str='llama-3.3-70b-versatile',
                     temperature: float = 0.3,
                     max_tokens: int = 512,

                     ):
    data=pd.read_json(path_data_to_eval,lines=True)
    if all_chunks is None:
        with open('chunks/all_fandom_chunks.pkl', 'rb') as f:
            all_chunks=pickle.load(f)

    results=data.apply(lambda x: generate_response(x['question'],
                                                              all_chunks=all_chunks,
                                                              name_model=name_model,
                                                              temperature=temperature,
                                                              max_tokens=max_tokens,

                                                              return_time=True,
                                                              is_oracle_retriever=True,
                                                              relevant_chunk_ids=x['relevant_chunk_ids']), axis=1)
    data['llm_answer'], data['generation_time'], data['e2e_latency'], data['relevant_chunks']=zip(*results)
    return data

if __name__=='__main__':
    faiss_index, all_chunks=load_chunks(path_faiss_embed_chunks='bm3/faiss/all_fandom_bge-m3.faiss')
    #print(generate_response(query='How was the third Mizukage in Naruto?',retriever_type='bm25'))
    #print(generate_response(query='Кто более сильный Наруто или Сасуке?',retriever_type='faiss',show_time=True,faiss_index=faiss_index,threshold=0.53))
    #print(generate_response(query='When will produce 200 episods of anime Hunter?',retriever_type='bm25'))
    #print(generate_response(query='What will cost of the dollar in nex dday?',retriever_type='bm25'))
    data=oracle_retriever('eval_set_v3_clean.jsonl',all_chunks)
    print(data.info())