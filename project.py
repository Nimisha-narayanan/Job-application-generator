import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Project:
    def __init__(self, file_path=r"C:\DL projects\cold_email_generator_for_applicants\my_projects.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)

        # Vectorizer for text similarity
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tech_vectors = self.vectorizer.fit_transform(self.data["Techstack"])


    def query_links(self, skills, top_k=2):
        if isinstance(skills, list):
            skills = " ".join(skills)

        skill_vector = self.vectorizer.transform([skills])
        similarities = cosine_similarity(skill_vector, self.tech_vectors).flatten()

        top_indices = similarities.argsort()[-top_k:][::-1]
        return self.data.iloc[top_indices]["Links"].tolist()

    


