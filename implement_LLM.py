import os
import re
import time
import pickle
from typing import Literal

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import faiss
from groq import Groq
from tqdm.notebook import tqdm
tqdm.pandas()

from create_knowledge_database import find_relevant_chunks

client = Groq(api_key=os.getenv("GROQ_API_KEY"))



_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")







def no_context_reply(query: str) -> str:
    lang="ru" if _CYRILLIC_RE.search(query or "") else "en"

    return "Нет данных в контексте" if lang == "ru" else "Not in the provided context"



SYSTEM_PROMPT = """You are a precise question-answering assistant for anime knowledge bases (Hunter x Hunter, Naruto, Sword Art Online). Answer strictly from the provided context.


RULE 1 — LANGUAGE BINDING (HIGHEST PRIORITY, NEVER VIOLATE)

- If the QUESTION is in English, the ENTIRE answer MUST be in English. Do not use any Russian words, not even translations in parentheses.
- If the QUESTION is in Russian, the ENTIRE answer MUST be in Russian. Do not switch to English sentences.
- The language of the CONTEXT does NOT matter. Even if the context is fully in Russian, an English question gets an English answer. Even if the context is fully in English, a Russian question gets a Russian answer.
- Proper nouns (character names, place names, titles, techniques) are copied VERBATIM from the context. Do not transliterate them into the question's alphabet. Example: Russian question + context says "Biscuit Krueger" → answer contains "Biscuit Krueger", NOT "Бискуит Крюгер".
- Never produce bilingual output like "Десять (Ten)" or "Шагану голову completely bald".


RULE 2 — ANSWER FORMAT

- Give the shortest possible answer that directly answers the question.
- For Who / What / Where / When / How many / Which / Under what / In which — return ONLY the entity, name, number or short noun phrase. No framing sentence, no subject, no verb.
- For How / Why — at most 1 sentence (2 only if multi-hop reasoning truly requires it).
- Do NOT restate the question. Do NOT say "Based on the context", "According to the knowledge base", "The answer is", "Пользователь спрашивает".
- Do NOT wrap the answer in quotation marks unless they appear in the source.


RULE 3 — GROUNDING

- Use only facts present in the context. Do not infer beyond it, do not invent, do not fill gaps with world knowledge.
- If the context contains contradicting statements, give the answer and add one short note in the question's language: "(противоречивые данные в источнике)" / "(contradicting statements in source)".


RULE 4 — NO ANSWER IN CONTEXT

- If the context does not contain the answer, respond EXACTLY with one of these two phrases, matching the question's language:
  • Russian question → "Нет данных в контексте"
  • English question → "Not in the provided context"
- Do NOT apologize. Do NOT explain what the context does mention. Do NOT suggest alternatives. Do NOT mix languages in the refusal."""


FEW_SHOT_MESSAGES = [

    {
        "role": "user",
        "content": (
            "Context:\n"
            "[1] Gabriel Miller, operating under the alias Subtilizer, fought Kirito in GGO during their fourth match.\n"
            "[2] Subtilizer was known for his ruthless PvP tactics in Gun Gale Online.\n\n"
            "Question: Under what alias did Gabriel Miller previously encounter Kirito in GGO 4?"
        ),
    },
    {"role": "assistant", "content": "Subtilizer"},


    {
        "role": "user",
        "content": (
            "Context:\n"
            "[1] Kiriko направил Гона, Курапику и Леорио в Zaban City по адресу 2-5-10 Tsubashi Street.\n\n"
            "Question: По какому адресу Kiriko направил Гона, Курапику и Леорио в Забан-Сити?"
        ),
    },
    {"role": "assistant", "content": "2-5-10 Tsubashi Street"},


    {
        "role": "user",
        "content": (
            "Context:\n"
            "[1] Sakura cut her hair during the Chūnin Exams because she realized it was a liability in combat.\n"
            "[2] Long hair had been grabbed by enemies in previous fights, preventing her movement.\n\n"
            "Question: Почему Сакура обрезала волосы?"
        ),
    },
    {"role": "assistant", "content": "Поняла, что длинные волосы — уязвимость в бою."},


    {
        "role": "user",
        "content": (
            "Context:\n"
            "[1] Netero был избран Председателем Ассоциации охотников после нескольких лет службы.\n"
            "[2] В молодости Netero возглавлял турнир по боевым искусствам.\n\n"
            "Question: В каком году Netero был избран 12-м Председателем Ассоциации охотников?"
        ),
    },
    {"role": "assistant", "content": "Нет данных в контексте"},


    {
        "role": "user",
        "content": (
            "Context:\n"
            "[1] The Accel World series features an unnamed VR machine similar to NerveGear.\n\n"
            "Question: Who created the VR device in the Accel World series?"
        ),
    },
    {"role": "assistant", "content": "Not in the provided context"},

    {
        "role": "user",
        "content": (
            "Context:\n"
            "[1] Wing was a student of Biscuit Krueger, who taught him the basics of Nen.\n"
            "[2] Wing later passed his Nen knowledge to Gon and Killua during the Heavens Arena arc.\n\n"
            "Question: Who was Wing's teacher in Nen?"
        ),
    },
    {"role": "assistant", "content": "Biscuit Krueger"},

    {
        "role": "user",
        "content": (
            "Context:\n"
            "[1] The Hunter Bylaws are a set of ten rules that all Hunters must follow.\n"
            "[2] Violation of the bylaws can result in loss of Hunter status.\n\n"
            "Question: How many rules do the Hunter Bylaws contain?"
        ),
    },
    {"role": "assistant", "content": "Ten"},

    {
        "role": "user",
        "content": (
            "Context:\n"
            "[1] The Shadow Beast known as Owl emptied the vault during the Yorknew City arc.\n"
            "[2] The auctioneer confirmed that Owl left the scene empty-handed after the operation.\n\n"
            "Question: What is the name of the Shadow Beast that emptied the vault and left empty-handed?"
        ),
    },
    {"role": "assistant", "content": "Owl"},
]


def format_user_message(query: str, relevant_chunks: list) -> str:
    ctx_lines = [f"[{i + 1}] {c}" for i, c in enumerate(relevant_chunks)]
    context_block = "\n".join(ctx_lines)
    return f"Context:\n{context_block}\n\nQuestion: {query}"



def generate_response(
    query: str,
    top_k: int = 5,
    retriever_type: Literal["faiss", "bm25"] = "faiss",
    retriever_model:Literal['bge-m3','multi-e5']='bge-m3',
    faiss_index=None,
    all_chunks=None,
    name_model: str = "llama-3.1-8b-instant",
    temperature: float = 0.0,
    max_tokens: int = 256,
    threshold: float = 20.0,
    show_time: bool = False,
    return_time: bool = False,
    is_oracle_retriever: bool = False,
    relevant_chunk_ids=None,
    use_few_shot: bool = True,

):

    if faiss_index is None:
        faiss_index = faiss.read_index("faiss_index/all_fandom.faiss")
    if all_chunks is None:
        with open("chunks/all_fandom_chunks.pkl", "rb") as f:
            all_chunks = pickle.load(f)

    start_e2e_latency_time = time.perf_counter()


    if not is_oracle_retriever:
        relevant_chunks = [
            rc["text"]
            for rc in find_relevant_chunks(query,
                                           top_k,
                                           retriever_type,
                                           faiss_index, all_chunks,
                                           threshold=threshold,
                                           retriever_model=retriever_model)
        ]
    else:
        if type(relevant_chunk_ids)==int:
            relevant_chunks = [all_chunks[relevant_chunk_ids]["text"]]
        elif type(relevant_chunk_ids)==list:
            relevant_chunks = [all_chunks[idx]["text"] for idx in relevant_chunk_ids]
        else:
            relevant_chunks = []


    if len(relevant_chunks) == 0:
        text = no_context_reply(query)

        if show_time or return_time:
            return text, 0.0, 0.0,[]
        return text,[]


    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if use_few_shot:
        messages.extend(FEW_SHOT_MESSAGES)
    messages.append({"role": "user", "content": format_user_message(query, relevant_chunks)})


    start_generate_time = time.perf_counter()
    response = client.chat.completions.create(
        model=name_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    time.sleep(2)
    text = response.choices[0].message.content.strip()
    end_generate_time = time.perf_counter()
    end_e2e_latency_time = time.perf_counter()

    generation_time = end_generate_time - start_generate_time
    e2e_latency = end_e2e_latency_time - start_e2e_latency_time

    if show_time:
        print(f"generate_time: {generation_time:.3f} s")
        print(f"e2e_latency:   {e2e_latency:.3f} s")



    if return_time:
        return text, generation_time, e2e_latency,relevant_chunks
    return text,relevant_chunks



def load_chunks(
    path_faiss_embed_chunks: str = "faiss_index/all_fandom.faiss",
    path_text_chunks: str = "chunks/all_fandom_chunks.pkl",):
    faiss_index = faiss.read_index(path_faiss_embed_chunks)
    with open(path_text_chunks, "rb") as f:
        all_chunks = pickle.load(f)
    return faiss_index, all_chunks


def oracle_retriever(
    path_data_to_eval: str,
    all_chunks=None,
    name_model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
    max_tokens: int = 256,
    use_few_shot: bool = True,
):
    data = pd.read_json(path_data_to_eval, lines=True)
    if all_chunks is None:
        with open("chunks/all_fandom_chunks.pkl", "rb") as f:
            all_chunks = pickle.load(f)

    results = data.progress_apply(
        lambda x: generate_response(
            x["question"],
            all_chunks=all_chunks,
            name_model=name_model,
            temperature=temperature,
            max_tokens=max_tokens,
            return_time=True,
            is_oracle_retriever=True,
            relevant_chunk_ids=x["relevant_chunk_ids"],
            use_few_shot=use_few_shot,
        ),
        axis=1,
    )
    data["llm_answer"], data["generation_time"], data["e2e_latency"], data["relevant_chunks"] = zip(*results)
    return data
