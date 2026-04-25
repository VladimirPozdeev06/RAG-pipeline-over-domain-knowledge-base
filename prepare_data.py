import xml.etree.ElementTree as ET
import mwparserfromhell
import re
import json
from langdetect import DetectorFactory,detect_langs,LangDetectException
import pymorphy3

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()
stopwords_en = set(stopwords.words('english'))
morph = pymorphy3.MorphAnalyzer()


NAMESPACE='{http://www.mediawiki.org/xml/export-0.11/}'
def clean_wiki_text(text):
    clean_text=mwparserfromhell.parse(text).strip_code().strip()
    clean_text=re.sub(r'\n{3,}','\n\n',clean_text)
    clean_text=re.sub(r'Category:.*','',clean_text)
    clean_text = re.sub(r'\n[a-z]{2}:[^\n]+', '', clean_text)
    clean_text = re.sub(r'Hunter × Hunter[^.\n]*', '', clean_text)
    clean_text = re.sub(r'thumb\|.*?\n', '', clean_text)
    if len(clean_text.strip())>=100:
        return clean_text
    return None

def load_wiki_fandom(path_to_xml:str,path_to_json_file:str):
    tree = ET.parse(path_to_xml)
    root = tree.getroot()

    with open(path_to_json_file, 'w', encoding='utf-8') as f:
        for page in root.findall(f'{NAMESPACE}page'):
            title = page.find(f'{NAMESPACE}title')
            text = page.find(f'.//{NAMESPACE}text')
            redirect = page.find(f'{NAMESPACE}redirect')
            if redirect is None and text is not None and text.text is not None:
                clean_text = clean_wiki_text(text.text)
                if clean_text is not None:
                    result = {'title': title.text, 'clean_text': clean_text}

                    f.write(json.dumps(result, ensure_ascii=False) + '\n')




def detect_english_text(text:str,min_confidence:float):
    DetectorFactory.seed = 0
    if len(text.strip()) <=3:
        return False
    try:
        langs=detect_langs(text)
        for lang in langs:
            if lang.lang=='en' and lang.prob>=min_confidence:
                return True
        return False
    except LangDetectException as e:
        return False
def lemmatize_en(text: str) -> list[str]:
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stopwords_en and len(t) > 1
    ]

def lemmatize_ru(text: str) -> list[str]:
    tokens = re.findall(r'\b[а-яёА-ЯЁ]+\b', text.lower())
    return [morph.parse(t)[0].normal_form for t in tokens]
if __name__ == '__main__':
    load_wiki_fandom('hunterxhunter.fandom.com-20260413-wikidump/hunterxhunter.fandom.com-20260413-current.xml',
                     'hunter_parsed_pages.jsonl')

    load_wiki_fandom('naruto.fandom.com-20260413-wikidump/naruto.fandom.com-20260413-current.xml',
                  'naruto_parsed_pages.jsonl')

    load_wiki_fandom('swordartonline.fandom.com-20260413-wikidump/swordartonline.fandom.com-20260413-current.xml',
                     'sao_parsed_pages.jsonl')
