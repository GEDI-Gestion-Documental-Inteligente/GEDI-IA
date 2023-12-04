from pymongo import MongoClient
from datetime import datetime, timedelta
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from typing import Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain.document_loaders import PyPDFLoader

load_dotenv()

semipath = "uri"

def query_mongo_atlas(query):
    """Consulta MongoDB Atlas y devuelve resultados."""
    client = MongoClient(
        "mongodb+srv://walterlul:milonga123@gedi-cluster.nxt2obp.mongodb.net/?retryWrites=true&w=majority")
    db = client.test
    collection = db["nodes"]
    results = list(collection.find(query))
    first_element = results[0]
    document_name = first_element['path']
    completed_path = document_name + semipath

    loader = PyPDFLoader(completed_path)
    pages = loader.load_and_split()

    return pages

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
    query: dict = Field(description="The query to execute on the 'nodes' collection")

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

tools = [MongoDBQueryTool()]

agent = initialize_agent(
    tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

agent.run(
    "Busca el documento con el nombre 'FUNDAMENTOS SEGURIDAD' y muestramelo"
)