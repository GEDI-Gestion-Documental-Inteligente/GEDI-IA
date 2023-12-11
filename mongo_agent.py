from pymongo import MongoClient
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from typing import Type
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import pandas as pd
import io
from langchain.document_loaders import PyPDFLoader
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from create_database import split_text

load_dotenv()
app = FastAPI()
CHROMA_PATH = "chroma"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    query: str


def query_mongo_atlas(query):
    """Consulta MongoDB Atlas y devuelve resultados."""
    client = MongoClient(
        "mongodb+srv://walterlul:milonga123@gedi-cluster.nxt2obp.mongodb.net/?retryWrites=true&w=majority"
    )
    db = client.test
    collection = db["nodes"]
    results = list(collection.find({"name": query}))
    if not results:
        return "No se encontraron resultados"
    first_element = results[0]
    document_name = first_element["path"]

    file = requests.get(f"https://dvcm270k-4000.brs.devtunnels.ms{document_name}")

    if document_name.endswith(".csv"):
        print("es un archivo CSV")
        data = file.content.decode("utf-8")
        df = pd.read_csv(io.StringIO(data))
        return df
    elif document_name.endswith(".pdf"):
        print("es un archivo PDF")
        print(document_name)
        loader = PyPDFLoader(f"https://dvcm270k-4000.brs.devtunnels.ms{document_name}")
        documents = loader.load()
        chunk = split_text(documents)
        db = Chroma.from_documents(
            chunk, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
        )
        db.persist()
        if not isinstance(query, str):
            query = str(query)
        doc = db.similarity_search(query)
        return doc[0]
    else:
        return "Unsupported file type"


class MongoQueryInput(BaseModel):
    """Input for the MongoDB query."""

    name: str = Field(
        description="The name of the document to find in the 'nodes' collection"
    )


class MongoDBQueryTool(BaseTool):
    name = "query_mongo_atlas"
    description = "Finds a document whit .pdf or .csv"
    args_schema: Type[BaseModel] = MongoQueryInput

    def _run(self, name: str):
        return query_mongo_atlas(name)

    def _arun(self, name: str):
        raise NotImplementedError("Asynchronous execution not supported")


llm = ChatOpenAI(model="gpt-4", temperature=0)

tools = [MongoDBQueryTool()]

agent_mongo = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
