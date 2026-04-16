import os
from dotenv import load_dotenv
load_dotenv()

from groq import Groq
import faiss
import pickle
from create_knowledge_database import find_relevant_chunks
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)
def generate_response(query:str,
                      top_k:int=5,
                      faiss_index=None,all_chunks=None,
                      name_model:str='llama-3.3-70b-versatile',
                      temperature:float=0.3,
                      max_tokens=512):
    if faiss_index is None:
        faiss_index = faiss.read_index('faiss_index/all_fandom.faiss')
    if all_chunks is None:
        with open('chunks/all_fandom_chunks.pkl', 'rb') as f:
            all_chunks=pickle.load(f)

    relevant_chunks=[rc['text'] for rc in find_relevant_chunks( query, top_k, faiss_index,all_chunks)]
    response=client.chat.completions.create(
       model=name_model,
       messages=[
           {
               'role':'system',
               'content': 'You are expert in anime topic. Answer on question using provided content. You should obligatory use it and notify if he can not find an answer or information can be incomplete. Moreover if you find opposite information you should also say about it. Do not lye and imagine. Response only on query question.'
           },
           {
               'role':'user',
               'content': f'I have a question: {query}. Give me an answer base on next knowledge base: {relevant_chunks}'
           }
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    text=response.choices[0].message.content
    return text

if __name__=='__main__':
    print(generate_response('How was the first Mizukage in Naruto?'))
    print(generate_response('How is more powerful Hisoka or Kirito?'))
    print(generate_response('What will cost of the dollar in nex dday?'))