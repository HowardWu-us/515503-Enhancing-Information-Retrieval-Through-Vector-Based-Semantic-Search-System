# %%
from gensim.models import Word2Vec, KeyedVectors

# %%
# 下載預訓練好的 Google News Word2Vec 模型
# 需要到 https://github.com/mmihaltz/word2vec-GoogleNews-vectors 下載模型
model = KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)

# %%
# 獲取詞彙空間向量
vector = model['king']
print(vector)

# %%
# 計算詞語相似性
similarity = model.similarity('apple', 'apples')
print(f"Similarity between 'apple' and 'apples': {similarity}")

# %%
