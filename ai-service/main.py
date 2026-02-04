from dotenv import load_dotenv

load_dotenv()

# from langchainhub import pull
# from langchainhub import hub
# from langchain.agents.react.agent import create_agent
# from langchain.agents.react.prompt import REACT_PROMPT
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
llm = ChatGroq(temperature=0, model="llama-3.1-8b-instant")
agent = create_agent(model=llm, tools=tools)


# agent = create_agent(model = llm, tools = tools)
def main():
    print("Hello from LangChain!")
    result = agent.invoke(
        {"messages": [("user", "Who is the president of the United States?")]}
    )
    print(result)


if __name__ == "__main__":
    main()
