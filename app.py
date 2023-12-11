from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from help import MongoDBQueryTool
# Importa aquí tus otros módulos y funciones necesarias
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.agents.agent_toolkits import create_retriever_tool
from dotenv import load_dotenv
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool

# from langchain.agents import initialize_agent
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent

load_dotenv()

# * Retriever Tool Logic
embedding_function = OpenAIEmbeddings()

CHROMA_PATH = "chroma"

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

retriever = db.as_retriever()

tool_retriever = create_retriever_tool(
    retriever,
    "search_documents",
    "Searches and returns documents.",
)
llm = OpenAI(temperature=0)

# * Python Tool
python_repl = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# * SQL Tool
# db = SQLDatabase.from_uri("sqlite:///Chinook_Sqlite.sqlite")
# toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# sql_agent = create_sql_agent(
#     llm=ChatOpenAI(temperature=0),
#     toolkit=toolkit,
#     verbose=True,
#     agent_type=AgentType.OPENAI_FUNCTIONS,
# )

# * All Tools
tools = [
    Tool(
        name="Python_RELP",
        func=python_repl.run,
        description="useful for when you need to use python to answer a question. You should input python code",
    ),
    #     Tool(
    #     name="Sql_Agent",
    #     func=sql_agent.run,
    #     description="Util para consultar datos en la base de datos",
    # ),
]

# * Retriever Tool Add
tools.append(MongoDBQueryTool())
tools.append(tool_retriever)

llm = ChatOpenAI(temperature=0)

agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)

# ! This is needed for both the memory and the prompt
memory_key = "history"

memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

system_message = SystemMessage(
    content=(
        "Do your best to answer the questions. "
        "Feel free to use any tools available to look up "
        "relevant information, only if necessary"
        "If query have .pdf or .csv use MongoDBQueryTool"
    )
)

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
)

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True
)


# Modelo Pydantic para la entrada de datos
class QueryData(BaseModel):
    query: str


# Inicializar FastAPI y CORS
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
def process_query(query_data: QueryData):
    try:
        logging.debug(f"Received query: {query_data.query}")
        result = agent_executor({"input": query_data.query})
        return result["output"]
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)