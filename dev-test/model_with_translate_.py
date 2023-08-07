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


md=['all-MiniLM-L6-v2','intfloat/e5-large-v2']
embedder = SentenceTransformer(md[0])


def translate_text(text, target_language='en'):
    from_code = "ru"
    to_code = "en"
    translatedText = argostranslate.translate.translate(text, from_code, to_code)
    return translatedText

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
corpus_org = [x+" "+y for x, y in zip(corpus1, corpus2)]
corpus = [translate_text(x, target_language='en') for x in corpus_org]

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)


def test():
    while True: 
        query_main=input("Ввод:")
        query=translate_text(query_main, target_language='en')
        top_k = min(num_res, len(corpus))
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        print("\n\n======================\n\n")
        print("Предложение:", query_main)
        print("\nТоп 10 подходящих:")

        for score, idx in zip(top_results[0], top_results[1]):
            if score>0.15:
                print(idx,corpus1[idx], "(Score: {:.4f})".format(score))
                print(corpus2[idx])
    