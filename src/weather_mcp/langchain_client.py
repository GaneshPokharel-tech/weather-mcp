import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleRateLimitError

# Load environment variables from .env
load_dotenv()


def print_message_content(content):
    """Print Gemini/LangChain message content cleanly."""

    if isinstance(content, str):
        print(content)
        return

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                print(block.get("text", ""))


async def main():
    # -------------------------------------------------
    # Configure Weather MCP server
    # -------------------------------------------------

    config = {
        "mcpServers": {
            "weather": {
                "command": "uv",
                "args": [
                    "run",
                    "mcp",
                    "run",
                    "src/weather_mcp/server.py",
                ],
            }
        }
    }

    # -------------------------------------------------
    # Create Gemini model
    # -------------------------------------------------

    model = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0,
    )

    # -------------------------------------------------
    # Connect LangChain to MCP server
    # -------------------------------------------------

    async with MCPAdapter(config) as adapter:
        # Discover MCP tools
        tools = await adapter.list_tools()

        print("Available MCP tools:")

        for tool in tools:
            print(f"- {tool.name}")

        # -------------------------------------------------
        # Create LangChain agent
        # -------------------------------------------------

        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=(
                "You are a weather assistant. "
                "Use the available MCP weather tools for weather questions. "
                "Do not invent weather information. "
                "If a location is ambiguous, do not guess. "
                "Ask the user for a more specific location, "
                "preferably city and country. "
                "Use information from earlier messages in the conversation "
                "when the user provides clarification."
            ),
        )

        # -------------------------------------------------
        # Conversation history
        # -------------------------------------------------

        conversation_history = []

        print("\nWeather Assistant")
        print("Type 'exit' or 'quit' to stop.\n")

        # -------------------------------------------------
        # Multi-turn conversation loop
        # -------------------------------------------------

        while True:
            question = input("You: ").strip()

            # Stop program
            if question.lower() in {"exit", "quit"}:
                print("Session ended.")
                break

            # Ignore empty input
            if not question:
                continue

            # Add user message to conversation history
            conversation_history.append(HumanMessage(content=question))

            # Remember number of messages before agent runs
            previous_message_count = len(conversation_history)

            # -------------------------------------------------
            # Run agent
            # -------------------------------------------------

            try:
                result = await agent.ainvoke(
                    {
                        "messages": conversation_history,
                    }
                )

            # -------------------------------------------------
            # Handle Gemini quota/rate-limit errors
            # -------------------------------------------------

            except GoogleRateLimitError:
                print("\nAI Error: Gemini API quota or rate limit " "has been reached.")
                print(
                    "Please wait for the quota to become available " "and try again.\n"
                )

                # Keep the conversation alive
                continue

            # -------------------------------------------------
            # Save updated conversation history
            # -------------------------------------------------

            conversation_history = result["messages"]

            # Messages created during this turn
            new_messages = conversation_history[previous_message_count:]

            # -------------------------------------------------
            # Show MCP tool calls
            # -------------------------------------------------

            for message in new_messages:
                tool_calls = getattr(
                    message,
                    "tool_calls",
                    None,
                )

                if tool_calls:
                    print("\nTool Call:")

                    for tool_call in tool_calls:
                        print(f"  Tool: {tool_call['name']}")
                        print(f"  Arguments: {tool_call['args']}")

            # -------------------------------------------------
            # Print final AI response
            # -------------------------------------------------

            final_message = conversation_history[-1]

            print("\nAI:")
            print_message_content(final_message.content)
            print()


if __name__ == "__main__":
    asyncio.run(main())
