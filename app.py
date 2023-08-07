from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token,
    get_jwt_identity
)

''''''
file_name="Тестовые данные.csv"
# from_code = "ru"
# to_code = "en"

num=200
num_res=100
min_score=0
data = ""

corpus = []

f=False
s=False
fs=True

corpus_embeddings = []
''''''

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = '{youre secret key}'
jwt = JWTManager(app)

CORS(app)

# def download_tr():
#     argostranslate.package.update_package_index()
#     available_packages = argostranslate.package.get_available_packages()
#     package_to_install = next(
#         filter(
#             lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
#         )
#     )
#     argostranslate.package.install_from_path(package_to_install.download())

# # def translate_text(text, source_language='auto', target_language='en'):
#     translator = Translator(service_urls=['translate.googleapis.com'])

#     try:
#         translated_text = translator.translate(text, src=source_language, dest=target_language)
#         return translated_text.text
#     except Exception as e:
#         # print(f"Translation error: {e}")
#         return ""

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


with app.app_context():
    print(1)
    file_name="Тестовые данные.csv"
    from_code = "ru"
    to_code = "en"

    num=200
    num_res=100
    min_score=0
    data = pd.read_csv(file_name,delimiter=';').head(num)


    md=['all-MiniLM-L6-v2','intfloat/e5-large-v2','lsanochkin/setfit-rubert-intent']
    embedder = SentenceTransformer(md[2])

    corpus1 = data["title"].tolist()
    corpus2 = data["description"].tolist()
    if fs:
        corpus_org = [x+" "+y for x, y in zip(corpus1, corpus2)]
    if f:
        corpus_org = [x for x in zip(corpus1, corpus2)]
    if s:
        corpus_org = [y for y in zip(corpus1, corpus2)]
    corpus = [x for x in corpus_org]

    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)




def many_search(tags,st,inp):
        
    pr=[0 for x in range(0,num)]


    if inp!="" or inp!=None:

        query=inp
        top_k = min(num, len(corpus))
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        # print("\n\n======================\n\n")
        # print("Предложение:", query_main)
        # print("\nТоп 10 подходящих:")
        for score, idx in zip(top_results[0], top_results[1]):
            a="{}".format(idx)
            a=int(a)
            b="{:.4f}".format(score)
            b=float(b)*100
            # print(pr[a])
    
            pr[a]=pr[a]+b

    # print(tags,st)
    for x in tags:

        query=x
        top_k = min(num, len(corpus))
        query_embedding = embedder.encode(query.lower(), convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        # print("\n\n======================\n\n")
        # print("Предложение:", query_main)
        # print("\nТоп 10 подходящих:")
        for score, idx in zip(top_results[0], top_results[1]):
            a="{}".format(idx)
            a=int(a)
            b="{:.4f}".format(score)
            b=float(b)*100
            if x[0]=="-":
                pr[a]=float(pr[a])-float(b/100)*float(st)
            else:
                pr[a]=float(pr[a])+float(b/100)*float(st)

    res=[]


    for x in range(0,num):
                 res.append({
                'id':x,
                'title': corpus1[x],
                'description': corpus2[x],
                'prc':pr[x]
                })
    return res

# def read_csv_to_text_array(input_file):
#     text_array = []

#     with open(input_file, mode='r') as file:
#         reader = csv.reader(file)

#         for row in reader:
#             if row:
#                 text_array.append(row[0])

#     return text_array

# input_file = "TSENG.csv"

# text_array = read_csv_to_text_array(input_file)

# def test():
#     while True: 
#         query_main=input("Ввод:")
#         query=translate_text(query_main, target_language='en')
#         top_k = min(num_res, len(corpus))
#         query_embedding = embedder.encode(query, convert_to_tensor=True)
#         cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
#         top_results = torch.topk(cos_scores, k=top_k)

#         print("\n\n======================\n\n")
#         print("Предложение:", query_main)
#         print("\nТоп 10 подходящих:")

#         for score, idx in zip(top_results[0], top_results[1]):
#             if score>0.15:
#                 print(idx,corpus1[idx], "(Score: {:.4f})".format(score))
#                 print(corpus2[idx])

users = {
    'admin1': {
        'username': 'admin1',
        'password': 'password1'
    }
}


def search(query_main):
        
        res=[]

        query=query_main
        top_k = min(num_res, len(corpus))
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        # print("\n\n======================\n\n")
        # print("Предложение:", query_main)
        # print("\nТоп 10 подходящих:")
        for score, idx in zip(top_results[0], top_results[1]):
            a="{}".format(idx)
            a=int(a)
            b="{:.4f}".format(score)
            b=float(b)*100
            if score>min_score:
                 res.append({
                'id':a,
                'title': corpus1[idx],
                'description': corpus2[idx],
                'prc':b
                })
                 
        return res

@app.route('/data', methods=['GET'])
@jwt_required()
def get_data():

    current_user = get_jwt_identity()
    data_from_frontend = request.args.get('inp')
    print(data_from_frontend)

    res=[]
    if data_from_frontend==None or data_from_frontend=='':
        for x in range(0,len(corpus1)):
            # print(x)
            res.append({
            'id':x,
                'title': corpus1[x],
                'description': corpus2[x],
                'prc':0
                })
    else:
        res=search(data_from_frontend)
        
    return jsonify(res)


@app.route('/data/tags', methods=['GET'])
@jwt_required()
def get_data_tags():

    current_user = get_jwt_identity()
    inp = request.args.get('inp')
    tags = request.args.get('tags').split('||')
    tag_st = request.args.get('tag_st')
    
    # print(inp)

    res=many_search(tags,tag_st,inp)
        
    return jsonify(res)


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"message": "Неверные данные для аутентификации"}), 400
    
    user = users.get(username)
    if user and user['password'] == password:
        access_token = create_access_token(identity=username)
        return jsonify({"access_token": access_token}), 200
    else:
        return jsonify({"message": "Неверное имя пользователя или пароль"}), 401


if __name__ == '__main__':

    app.run(debug=True)

