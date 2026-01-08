import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from project import Project
from utils import clean_text


def create_streamlit_app(llm, project, clean_text):
    st.title("📧 Job Application Generator")
    url_input = st.text_input(
        "Enter a Job Posting URL:",
        value="https://technopark.in/job-details/26395?job=Junior%20AI%20Engineer"
    )
    submit_button = st.button("Submit")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)

        
            jobs = llm.extract_jobs(data)

            for job in jobs:
                skills = job.get("skills", [])
                links = project.query_links(skills)

                # 1️⃣ Generate application email
                email = llm.write_mail(job, links)

                # 2️⃣ Generate mock interview questions
                questions = llm.generate_mock_questions(job)

                # ---- UI OUTPUT ----
                st.subheader("📧 Job Application Email")
                st.code(email, language="markdown")

                st.subheader("🎯 Mock Interview Questions")
                st.write(questions)

        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    project = Project()

    st.set_page_config(
        layout="wide",
        page_title=" Job Application Generator",
        page_icon="📧"
    )

    create_streamlit_app(chain, project, clean_text)
