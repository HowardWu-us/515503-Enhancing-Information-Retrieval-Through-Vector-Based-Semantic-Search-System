#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# In[2]:


class Article:
    article_url: str
    original_title: str
    translated_title: str
    
    def __init__(self, url: str, title: str) -> None:
        self.article_url = url
        self.original_title = title

class ArticleCollection:
    articles: list[Article]

    def __init__(self):
        self.articles = []

    def add_article(self, article: Article):
        self.articles.append(article)

    def info(self,):
        print(f"There are {len(self.articles)} titles.")
        
    def display_titles(self):
        for article in self.articles:
            print(f"Original Title: {article.original_title}")
            print(f"URL: {article.article_url}")
            print("-" * 15, '\n')


# In[3]:


class ArticleCollectionFromUrl(ArticleCollection):
    urls: list[str] = []
    def __init__(self, urls=[]):
        self.urls = urls
        super().__init__()

    def fetch_articles(self,):
        for url in tqdm(self.urls):
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                for item in soup.find_all("li", class_="announcement-item"):
                    link_tag = item.find("h2").find("a")
                    title = link_tag.text.strip()
                    href = link_tag["href"]
                    self.add_article(Article(url=href, title=title))
            else:
                print(f"無法訪問網站，狀態碼：{response.status_code}")
        return len(self.articles)


# In[4]:


PAGES = 1
Articles = ArticleCollectionFromUrl()
Articles.urls = [f"https://www.cs.nycu.edu.tw/announcements?page={i}" for i in range(1, PAGES+1)]

Articles.fetch_articles()
Articles.info()


# In[5]:


from googletrans import Translator
translator = Translator()
# tn = translator.translate("【學士班】113學年度資工系學士班「畢業學分預審」作業公告(請於10/11前繳交)", dest="en").text
# print(tn)
print("Translate articles to English")
for i in tqdm(Articles.articles):
    i.translated_title = translator.translate(i.original_title, dest="en").text
    # i.translated_title = i.original_title


# In[6]:


MODEL = "all-mpnet-base-v2"
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 預處理文本
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]# 移除標點
    tokens = [word for word in tokens if word not in stop_words]# 除掉停用詞
    return tokens

class ArticleSearch:
    def __init__(self, articles, model_name=MODEL):
        self.articles = articles
        self.model = SentenceTransformer(model_name)
        self.article_titles = [article.translated_title for article in articles]
        self.article_vectors = self.model.encode(self.article_titles, batch_size=32, show_progress_bar=True)

    # 回傳前K個相似度最高的向量
    def Kth_max(self, arr, k=1):
        return np.argsort(-arr, axis=0)[:k]

    def cosine_similarity_custom(self, A, B):
        dot_product = np.dot(A, B)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        return dot_product / (norm_A * norm_B)

    def search(self, query):
        query_vector = self.model.encode([query])
        similarities = np.array([self.cosine_similarity_custom(query_vector, b)[0] for b in self.article_vectors])
        return similarities

    def get_suggestions(self, query, k=5):
        similarities = self.search(query)
        suggestions = self.Kth_max(similarities, k)
        return suggestions, similarities

    def print_suggestions(self, query, k=5):
        suggestions, similarities = self.get_suggestions(query, k)
        for index in suggestions:
            print(f"{index}\t{similarities[index]:.4f} : {self.article_titles[index]}")
            print(f"\t\t {self.articles[index].article_url}")
        print(f"最相似的文章是: {self.article_titles[suggestions[0]]}")
print("loading MODEL...")
article_search = ArticleSearch(Articles.articles)


# In[19]:


class TestCase:
    query: str
    target: str

    def __init__(self, query, target):
        self.query = query
        self.target = target
    
    def find_target_index(self):
        target_index: int = None
        for idx, i in enumerate(Articles.articles):
            if i.original_title.find(self.target) != -1:
                # print(idx, i.original_title)
                target_index = idx
        return target_index

    def get_score(self, article_search):
        ARTICLE_SIZE = len(Articles.articles)
        similarities = article_search.search(self.query)

        target_idx = self.find_target_index()
        if target_idx == None:
            return 0
        most_similar_idx = article_search.Kth_max(similarities, k=ARTICLE_SIZE)
        if np.where(most_similar_idx==target_idx)[0].size != 0:
            return 1/(np.where(most_similar_idx==target_idx)[0][0]+1)
            return (np.where(most_similar_idx==target_idx)[0][0]+1)
        else:
            return 0

test_cases: list[TestCase] = [
    TestCase("NVIDIA 研替", "NVIDIA 2025 研發替代役/實習開放職缺資訊"),
    TestCase("114甄試", "114學年度碩士班甄試入學第2階段備取生名單及報到注意事項"),
    TestCase("甄試名單", "114學年度碩士班甄試入學第1階段"),
    TestCase("AI競賽", "AI Junior Award 2025"),
    TestCase("書卷獎", "【學士班】112學年度第2學期書卷獎得獎名單公告"),
    TestCase("導師名單", "113.10.15更新【學士班】113學年度第一學期大學部導生名單"),
    TestCase("特殊選材", "114學年度資訊工程學系特殊選才招生公告"),
    TestCase("畢業學分", "【學士班】113學年度資工系學士班「畢業學分預審」作業公告(請於10/11前繳交)"),
    TestCase("校友 頒獎", "資訊人院刊- 資訊系友【交大日資工系友回娘家暨傑出系友頒獎典禮】"),
    TestCase("學士畢業", "【學士班】資訊工程學系畢業離系/離校作業公告"),
]


# In[8]:


def translate_test_query():
    print("Translate TestCases to English...")
    for i in tqdm(test_cases):
        try:
            i.query = translator.translate(text=i.query, dest="en").text
        except AttributeError:
            print(f"cannot convert \"{i.query}\" to English")


# In[9]:


# query = test_cases[-1].query
# target = test_cases[-1].target
# article_search.print_suggestions(query, k=10)
# print(query)
# print(target)
# print(Articles.articles[test_cases[-1].find_target_index()].translated_title)


# In[10]:


# def find_target_index(target):
#     target_index: int
#     for idx, i in enumerate(Articles.articles):
#         if i.original_title.find(target) != -1:
#             target_index = idx
#     return target_index
# Articles.articles[find_target_index("AI Junior Award 2025")].original_title


# In[11]:


def runtest():
    scores = sum([i.get_score(article_search=article_search) for i in test_cases])/len(test_cases)
    print(f"score: {scores:.3f}")


# In[12]:


translate_test_query()


# In[20]:


[i.get_score(article_search=article_search) for i in test_cases]


# In[21]:


act: str = ""
while True:
    act = input("action: ")
    if act == "quit":
        break
    if act == "runtest":
        runtest()
    if act == "search":
        print("-"*10)
        query = input(": ")
        lang = translator.detect(query).lang
        if query[:2] == "ch":
            lang = "ch"
            query = query[2:]
        else:
            lang = "en"

        if lang == "en":
            article_search.print_suggestions(query=query, k=5)
        else:
            print("translating...")
            translated_query = translator.translate(text=query, dest="en").text
            print(f"->{translated_query}")
            article_search.print_suggestions(query=translated_query, k=5)


# In[17]:


get_ipython().system('jupyter nbconvert --to script SearchCSWeb.ipynb')


# In[18]:


get_ipython().system('which python')

