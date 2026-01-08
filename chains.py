import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
        ### JOB DESCRIPTION:
        {job_description}

        ### INSTRUCTION:
        You are Nimisha N, a fresher applying for the above role.
        Your task is to write a professional and concise job application email to the hiring manager.

        Guidelines:
        - start with Dear Hiring Manager and introduce yourself as Nimisha N but do not add acadamic details
        - Clearly state that you are a fresher.
        - Express genuine interest in the role.
        - Briefly align your skills with the job requirements.
        - Include the most relevant GitHub project links from the following list to demonstrate your hands-on experience: {link_list}
        - Mention that your resume is attached.
        - Maintain a polite, confident, and professional tone.
        - Do NOT exaggerate or invent experience.
        - Do NOT include any preamble, explanation, or markdown formatting.

        The email must end with:
        Kind regards,
        Nimisha N
        Email: nimishan190501@gmail.com
        Phone: 7025704937

        ### EMAIL (NO PREAMBLE):
        """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content
    
    def generate_mock_questions(self, job):
        prompt_questions = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}

        ### INSTRUCTION:
        Based on the above job description, generate mock interview questions
        suitable for a fresher candidate.

        Guidelines:
        - Include 5 to 7 questions.
        - Mix technical, conceptual, and basic behavioral questions.
        - Focus on the core skills mentioned in the job description.
        - Keep questions clear and concise.
        - Do NOT provide answers.
        - Do NOT include any preamble or explanation.

        ### MOCK INTERVIEW QUESTIONS:
        """
    )

        chain_questions = prompt_questions | self.llm
        res = chain_questions.invoke({"job_description": str(job)})
        return res.content


if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))