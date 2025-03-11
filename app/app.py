from langchain_community.document_loaders import DirectoryLoader

# Load the documents 
directory = './docs/'
def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
print(len(documents))

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents,chunk_size=500,chunk_overlap=100):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

# store the splitte documnets in docs variable
docs = split_docs(documents)


# from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

# from langchain_community.embeddings import SentenceTransformerEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# using chromadb as a vectorstore and storing the docs in it
# from langchain_community.vectorstores import Chroma
db = Chroma.from_documents(docs, embeddings)

# Doing similarity search  using query
query = "I don't have coding skill, can I still enter?"
matching_docs = db.similarity_search(query)

print(matching_docs[0])

# from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

#load_dotenv()
#os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

def get_answer(q : str):
   
    template = """
     System:
        You are Builder's League Discord Chatbot, a Chatbot design specifically to answer question asked by participant relating the Builder's League Hackathon. 
        Do not put html tags in your response.
       
        {context}

        Question: {question}
        Helpful Answer:
    """
    prompt_template = PromptTemplate(template=template, input_variables=["question", "context"])
    repo_id = "google/gemma-2b-it"
    
    # llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64,"max_new_tokens":512})

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
   
    chain = RetrievalQA.from_chain_type(llm=llm, 
            chain_type="stuff", 
            retriever=db.as_retriever(), 
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
    )

    result = chain.invoke({"query": q})
    return result

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "http://0.0.0.0:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Question(BaseModel):
    question: str
@app.post("/question")
def question_handler(question: Question):
    answer = get_answer(question.question)
    response = {"result": answer["result"].replace("\n", "<br/> "), "query": answer["query"]}
    return response

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8080)