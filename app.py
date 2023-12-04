from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from dotenv import load_dotenv
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase


load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    llm = OpenAI(temperature=0)

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    retriever = db.as_retriever()

    python_repl = create_python_agent(
        llm=llm,
        tool=PythonREPLTool(),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    retriever_tool = create_retriever_tool(
        retriever,
        "Responder_Preguntas",
        "Para responder preguntas sobre el pdf y csv",
    )

    db = SQLDatabase.from_uri("sqlite:///Chinook_Sqlite.sqlite")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    sql_agent = create_sql_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )

    file_tool = "test"
    
    tools = [
        Tool(
            name="Python RELP",
            func=python_repl.run,
            description="useful for when you need to use python to answer a question. You should input python code",
        ),
        Tool(
            name="Sql Agent",
            func=sql_agent.run,
            description="Util para consultar datos en la base de datos",
        ),
    ]|

    tools.append(retriever_tool)

    agent = initialize_agent(
        tools=tools, llm=llm, verbose=True, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
    )

    result = agent.run('Segun el csv de planilla alumnos hazme una tabla con los alumnos con mas faltas')


    print(result)
    
if __name__ == "__main__":
    main()
