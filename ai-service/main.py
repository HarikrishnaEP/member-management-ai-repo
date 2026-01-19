from typing import List

from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

#from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from tavily import TavilyClient
from langchain_tavily import TavilySearch


# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query, max_results=3)

class Source(BaseModel):
    """Schema for source used by agent"""

    url:str = Field(..., description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(..., description="The agent answer to the query")
    sources: List[Source] = Field(default_factory=list, description="The list of sources used by the agent")

llm = ChatGroq(temperature=0,model="llama-3.1-8b-instant")
tools = [TavilySearch()]
agent = create_agent(model=llm,tools=tools, response_format = AgentResponse) # Java equivalent: Agent agent = AgentFactory.createAgent(llm, tools);


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="what is the capital of france?")})
    print(result)

if __name__ == "__main__":
    main()