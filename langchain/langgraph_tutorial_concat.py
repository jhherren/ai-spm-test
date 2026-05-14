# from https://langchain-ai.github.io/langgraph/agents/agents/#2-create-an-agent

from langgraph.prebuilt import create_react_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


my_agent = create_react_agent(
    model="anthr" + "opic:" + "clau" + "de-3-7-son" + "net-latest",
    tools=[get_weather],
    prompt="You are a helpful assistant",
)

# Run the agent
my_agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
