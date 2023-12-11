from pydantic import BaseModel, Field
from openai import OpenAI
from typing import Type
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from dotenv import load_dotenv

load_dotenv()


class VisionQueryInput(BaseModel):
    """Input for the vision query."""

    image_url: str = Field(description="URL of the image")


class VisionQueryTool(BaseTool):
    name = "vision_query_tool"
    description = "Uses OpenAI GPT-4 Vision to analyze images and generate text."
    args_schema: Type[BaseModel] = VisionQueryInput

    def _run(self, image_url: str):
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Explicame esta imagen"},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        return response.choices[0]

    def _arun(self, image_url: str, text: str):
        raise NotImplementedError("Asynchronous execution not supported")


# Example of how to use the VisionQueryTool in an agent
llm = ChatOpenAI(model="gpt-4", temperature=0)

tools = [VisionQueryTool()]

agent_vision = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

# agent.run(
#     "Explicame esta imagen 'https://th.bing.com/th/id/R.01b702022503910847aa48d2256ef0a6?rik=YbbCgFMsYs6Cxw&riu=http%3a%2f%2fimages4.fanpop.com%2fimage%2fphotos%2f19800000%2fkenny-south-park-19880778-1920-1200.jpg&ehk=Ez9Yifos7zOGkp0W93qjZBoUIVJmxvtUTJCRy4whQmc%3d&risl=&pid=ImgRaw&r=0'",
# )
