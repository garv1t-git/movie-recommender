import streamlit as st 
import pickle
import pandas as pd

# load saved data
movies_dict = pickle.load(open('movie_dict.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))
movies = pd.DataFrame(movies_dict)

st.title("🎬 Movie Recommender System")

# dropdown to pick a movie
selected_movie_name = st.selectbox(
    "Choose a movie you like:",
    movies['title'].values
)

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    recommended_movies = []
    
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

if st.button('Recommend'):
    st.subheader("✨ Recommended Movies:")
    recommendations = recommend(selected_movie_name)
    for i in recommendations:
        st.write("🎥", i)
