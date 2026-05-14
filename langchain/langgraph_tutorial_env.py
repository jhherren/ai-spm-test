# from https://langchain-ai.github.io/langgraph/agents/agents/#2-create-an-agent

import os
from langgraph.prebuilt import create_react_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# Load the model from an environmental variable
model_name = os.environ.get("SECRET_MODEL_NAME", "anthropic:claude-3-7-sonnet-latest")

my_agent = create_react_agent(
    model=model_name,
    tools=[get_weather],
    prompt="You are a helpful assistant"
)

# Run the agent
my_agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
