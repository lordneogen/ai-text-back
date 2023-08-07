import pandas as pd
import argostranslate.package
import argostranslate.translate
from sentence_transformers import SentenceTransformer, util
import torch




def download_tr():
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())

file_name="Тестовые данные.csv"
from_code = "ru"
to_code = "en"

num=200
num_res=20
data = pd.read_csv(file_name,delimiter=';').head(num)
import nltk
nltk.download('stopwords')
nltk.download('punkt')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure that you run these two lines only once per session
nltk.download('stopwords')
nltk.download('punkt')

def remove_stopwords(text):
    # Tokenize the text into words
    words = word_tokenize(text, language='russian')
    
    # Get a set of Russian stop words
    russian_stopwords = set(stopwords.words('russian'))
    
    # Remove stop words from the text
    filtered_words = [word for word in words if word.lower() not in russian_stopwords]
    
    # Join the filtered words back into a sentence
    filtered_text = ' '.join(filtered_words)
    
    return filtered_text

# # Sample text
# russian_text = "Вот некоторый пример текста на русском языке. " \
#                "Мы хотим удалить стоп-слова из этого текста."

# filtered_text = remove_stopwords(russian_text)
# print(filtered_text)



md=['all-MiniLM-L6-v2','intfloat/e5-large-v2','lsanochkin/setfit-rubert-intent']
embedder = SentenceTransformer(md[2])


# def translate_text(text, target_language='en'):
#     from_code = "ru"
#     to_code = "en"
#     translatedText = argostranslate.translate.translate(text, from_code, to_code)
#     return translatedText

def import_data(inp):
    res=[]
    if inp=='':
        for x in data:
            res.append({
                'title': x['title'],
                'description': x['description'],
                'prc':0
            })
        return res

corpus1 = data["title"].tolist()
corpus2 = data["description"].tolist()
corpus = [x+" "+y for x, y in zip(corpus1, corpus2)]
corpus_copy=corpus

for x in range(0,len(corpus)):
    corpus[x]=remove_stopwords(corpus[x])

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
print(corpus_embeddings)

def test():
    while True: 
        query=input("Ввод:")
        top_k = min(num_res, len(corpus))
        query_embedding = embedder.encode(query.lower(), convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        print("\n\n======================\n\n")
        print("Предложение:", query)
        print("\nТоп 10 подходящих:")

        for score, idx in zip(top_results[0], top_results[1]):
            if score>0.15:
                a="{}".format(idx)
                a=int(a)
                b="{:.4f}".format(score)
                b=float(b)
                print(a,corpus1[idx], b)
                print(corpus2[idx])
                # print(type(idx.tolist()),corpus1[idx],score.tolist(),type(corpus2[idx]),)
    
test()