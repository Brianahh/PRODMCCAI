import streamlit as st
import numpy as np
import faiss
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from sklearn.preprocessing import normalize

df=pd.read_csv("tmdb_5000_movies.csv")
df = df.dropna(subset=["overview"])
movie_des = df["overview"].values
titles = df["title"]


#API
api_key = "8ceee6a2"


#model
from sentence_transformers import SentenceTransformer
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
model = load_model() #load model


@st.cache_resource
def load_embeddings():
    return np.load("moviedes_embeddings.npy")
moviedesc_embedding = load_embeddings() #load embeddings


@st.cache_resource
def load_embeddingstitle():
    return np.load("movietitle_embeddings.npy")
movietitle_embedding = load_embeddingstitle() #load embeddings


def faisswork(embedding):
  dimension = embedding.shape[1]
  index = faiss.IndexFlatIP(dimension)
  index.add(embedding)
  return index



def findmov(query,index, top_k=5):
    query_vector = model.encode([query], convert_to_numpy=True)
    query_vector = normalize(query_vector)
    scores, indices = index.search(query_vector, top_k)
    return [(df['title'].iloc[i], df['overview'].iloc[i], scores[0][j]*100)
            for j, i in enumerate(indices[0])]





def get_poster(movie_title):
    url = f"https://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()

    if data.get('Response') == 'True':
        poster_url = data.get('Poster')
        if poster_url and poster_url != 'N/A':
            try:
                image = Image.open(BytesIO(requests.get(poster_url).content))
                return image
            except:
                pass #fail thì chạy phía dưới 

    return Image.open("noimage.png")




def show_movie(results):
    for title, overview, score in results:
        st.subheader(title)
        col1,col2 = st.columns(2)
        with col1:
            st.write(overview)
            st.write(f"Similarity: {score:.2f}%")
        with col2:
            container = st.container(border=False,height=460)
            container.image(get_poster(title))
        



st.title("Movie Recommender")
user_input = st.text_input("Enter a description or movie idea:")
mode = st.selectbox(
    "Find movies based on:",
    ("Overview","Title"),
)


if user_input:
    if mode == "Overview":
      index = faisswork(moviedesc_embedding)
    elif mode == "Title":
      index = faisswork(movietitle_embedding)

    results = findmov(user_input,index)
    movie_titles = results[0]
    show_movie(results)
    
    
