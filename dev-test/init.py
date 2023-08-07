from flask import Flask, request, jsonify

import pandas as pd
import argostranslate.package
import argostranslate.translate
from sentence_transformers import SentenceTransformer, util
import torch


file_name="Тестовые данные.csv"
from_code = "ru"
to_code = "en"

num=200
num_res=100
min_score=0
data = pd.read_csv(file_name,delimiter=';').head(num)


def download_tr():
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())

def translate_text(text, target_language='en'):
    from_code = "ru"
    to_code = "en"
    translatedText = argostranslate.translate.translate(text, from_code, to_code)
    return translatedText


download_tr()
corpus1 = data["title"].tolist()
corpus2 = data["description"].tolist()
corpus_org = [x+" "+y for x, y in zip(corpus1, corpus2)]
corpus = [translate_text(x, target_language='en') for x in corpus_org]

import csv

def write_text_to_csv(text_array, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        for text in text_array:
            writer.writerow([text])

text_array = corpus

output_file = "TSENG.csv"

write_text_to_csv(text_array, output_file)

