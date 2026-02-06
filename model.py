import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("data.csv", index_col=0)

similarity = cosine_similarity(data)

user_index = 0  # target user
scores = similarity[user_index]

recommended_user = scores.argsort()[-2]

print("Recommended similar user index:", recommended_user)