import tiktoken
import pandas as pd
enc = tiktoken.get_encoding('cl100k_base')
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:

    tokens = enc.encode(text)
    step=chunk_size-overlap
    chunks = []
    for i in range(0, len(tokens), step):
        chunk = enc.decode(tokens[i:i + chunk_size])
        if len(chunk) >= 100:
            chunks.append(chunk)

    return chunks
if __name__ == '__main__':
    data_hunter=pd.read_json('hunter_parsed_pages.jsonl',lines=True)
    print(data_hunter.info())
    text=data_hunter['clean_text'].iloc[865]
    print(text)
    chunks=chunk_text(text)
    print(len(chunks))
    print(chunks)