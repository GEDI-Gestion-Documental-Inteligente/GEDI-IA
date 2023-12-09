from pymongo import MongoClient
from datetime import datetime, timedelta
import yfinance as yf
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from typing import Type
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents import initialize_agent
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.regex import Regex
import pandas as pd
import io
from langchain.document_loaders import PyPDFLoader
import requests
from fastapi import FastAPI
from typing import Any
from fastapi.middleware.cors import CORSMiddleware
from create_database import generate_data_store, load_documents, save_to_chroma, split_text

load_dotenv()
app = FastAPI()
CHROMA_PATH = "chroma"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Permite todas las origins
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos
    allow_headers=["*"],  # Permite todos los headers
)


class Query(BaseModel):
    query: str


def query_mongo_atlas(query):
    """Consulta MongoDB Atlas y devuelve resultados."""
    client = MongoClient(
        "mongodb+srv://walterlul:milonga123@gedi-cluster.nxt2obp.mongodb.net/?retryWrites=true&w=majority")
    db = client.test
    collection = db["nodes"]
    results = list(collection.find(query))
    first_element = results[0]
    document_name = first_element['path']

    # Handle different file types
    file = requests.get(
        f"https://dvcm270k-4000.brs.devtunnels.ms{document_name}")

    if document_name.endswith('.csv'):
        print("es un archivo CSV")
        data = file.content.decode('utf-8')
        df = pd.read_csv(io.StringIO(data))
        return df
    elif document_name.endswith('.pdf'):
        print("es un archivo PDF")
        print(document_name)
        loader = PyPDFLoader(
            f"https://dvcm270k-4000.brs.devtunnels.ms{document_name}")
        documents = loader.load()
        chunk = split_text(documents)
        save_to_chroma(chunk)
        embedding_function = OpenAIEmbeddings()

        dbc = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        # Convert the query to a string if it's not already
        if not isinstance(query, str):
            query = str(query)

        # Use the query in the similarity_search function
        doc = dbc.similarity_search(query)
        result = doc[0].page_content
        return result
    else:
        return "Unsupported file type"



# ejemplo
# nombre_documento = "FUNDAMENTOS seguridad"
# consulta_regex = Regex(f".*{nombre_documento}.*", 'i')
# resultado = query_mongo_atlas({"name": consulta_regex})

# print(type(resultado))

# # print(resultado)
# first_element = resultado[0] # This is a dictionary

# document_id = first_element['id']  # Access the 'id' from the dictionary
# document_name = first_element['path']  # Access the 'name' from the dictionary

# print(document_name)

# Update the Pydantic model to only include the query
class MongoQueryInput(BaseModel):
    """Input for the MongoDB query."""
    query: dict = Field(
        description="The query to execute on the 'nodes' collection")


# Update the MongoDBQueryTool to match the new function signature
class MongoDBQueryTool(BaseTool):
    name = "query_mongo_atlas"
    description = "Performs queries to the 'nodes' collection in MongoDB Atlas."
    args_schema: Type[BaseModel] = MongoQueryInput

    def _run(self, query: dict):
        # As collection name is now hardcoded in the function, it's no longer needed as a parameter
        return query_mongo_atlas(query)

    def _arun(self, query: dict):
        raise NotImplementedError("Asynchronous execution not supported")


llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)
# db = Chroma(persist_directory=CHROMA_PATH,
#                     embedding_function=embedding_function)



tools = [MongoDBQueryTool()]

agent = initialize_agent(
    tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)


@app.post("/chat")
async def run_query(query: Query):
    result = agent.run(query.query)
    return {"result": result}
