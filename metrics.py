import numpy as np
def recall_k(list_of_relevant_chunks:list,list_of_chunks:list,top_k:int)->float:
    k_chunks=list_of_chunks[:top_k]
    number_relevant_chunks_in_top_k=len(set(k_chunks) &set(list_of_relevant_chunks))
    number_relevant_chunks=len(list_of_relevant_chunks)
    return round(number_relevant_chunks_in_top_k/ number_relevant_chunks,3)

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
        IDCG += (1 / np.log2(rank + 1))
        if chunk in list_of_relevant_chunks:
            DCG+=(1/np.log2(rank+1))
    return round(DCG/IDCG,3)

def MRR(list_of_relevant_chunks:list,list_of_chunks:list)->float:
    for rank,chunk in enumerate(list_of_chunks):
        if chunk in list_of_relevant_chunks:
            return 1/(rank+1)
    return 0.0

