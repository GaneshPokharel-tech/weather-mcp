import asyncio
import json
import subprocess
from typing import Annotated

from dotenv import load_dotenv
from langchain.mcp import MCPAdapter
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleRateLimitError
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Project Configuration
# ---------------------------------------------------------

PROJECT_REPO = "/Users/ganesh/projects/multi-mcp-agent"


# ---------------------------------------------------------
# System Prompt
# ---------------------------------------------------------

SYSTEM_PROMPT = f"""
You are a multi-tool assistant connected to multiple MCP servers.

You have four main capabilities:

1. WEATHER

Available weather tools:
- weather_get_current_weather
- weather_get_weather_forecast

Rules:
- Use weather_get_current_weather for current weather.
- Use weather_get_weather_forecast for forecasts.
- Never invent weather information.
- If a location is ambiguous, do not guess.
- Ask the user to provide a city and country.


2. GMAIL

Available Gmail tools:
- gmail_search_emails
- gmail_get_email
- gmail_create_draft

Rules:
- Use gmail_search_emails when the user wants to search,
  find, or list emails.
- Use gmail_get_email when the full content of a specific
  email is required.
- Use gmail_create_draft only when the user explicitly asks
  to create an email draft.
- Creating a draft does NOT send the email.
- Never claim that an email was sent.
- Never claim an email was found unless a Gmail tool
  returned it.
- Never claim a draft was created unless
  gmail_create_draft succeeded.

Recommended Gmail sequence:

search request:
gmail_search_emails

full email request:
gmail_search_emails
-> gmail_get_email

draft request:
gmail_create_draft


3. FETCH

Available Fetch MCP tool:
- fetch_fetch

Rules:
- Use fetch_fetch when the user asks to retrieve or read
  content from a URL.
- Use fetched content as the source of truth.
- Never invent webpage content.
- Do not fetch localhost, private network addresses,
  cloud metadata endpoints, or other internal network
  resources unless the user has a clear legitimate reason
  and explicitly requests it.
- Do not use the Fetch MCP server for network scanning.
- If a page is too long, fetch additional content only
  when needed.


4. GIT

The allowed Git repository is:

{PROJECT_REPO}

Git MCP tools have names beginning with:
- git_

Examples may include:
- git_git_status
- git_git_diff_unstaged
- git_git_diff_staged
- git_git_diff
- git_git_log
- git_git_add
- git_git_commit
- git_git_create_branch
- git_git_checkout
- git_git_reset

Rules:
- Use the Git MCP tools when the user asks about the local
  Git repository.
- Use this repository path when a repo_path is required:

  {PROJECT_REPO}

- You may perform read-only Git operations when requested,
  such as status, log, and diff.
- Do not stage files unless the user explicitly asks.
- Do not commit unless the user explicitly asks.
- Do not reset changes unless the user explicitly asks.
- Do not switch or create branches unless the user
  explicitly asks.
- Never invent Git repository state.
- Git MCP is for the local Git repository. It is not the
  GitHub API.


GENERAL TOOL RULES

- Select tools based on the user's actual request.
- Do not call unrelated tools.
- When multiple tools are required, call them in the
  correct sequence.
- Use tool results as the source of truth.
- Do not claim an operation succeeded unless the tool
  reported success.
"""


# ---------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ---------------------------------------------------------
# Print AI response cleanly
# ---------------------------------------------------------


def print_message_content(content):
    if isinstance(content, str):
        print(content)
        return

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                print(block.get("text", ""))


# ---------------------------------------------------------
# macOS Notification
# ---------------------------------------------------------


def notify_draft_created():
    """
    Show a macOS desktop notification after a Gmail
    draft is successfully created.
    """

    subprocess.run(
        [
            "osascript",
            "-e",
            (
                "display notification "
                '"Gmail draft created successfully." '
                'with title "Multi MCP Agent"'
            ),
        ],
        check=False,
    )


# ---------------------------------------------------------
# Parse Tool Result
# ---------------------------------------------------------


def parse_tool_result(content):
    """
    Try to convert ToolMessage content into a dictionary.
    """

    if isinstance(content, dict):
        return content

    if isinstance(content, str):
        try:
            result = json.loads(content)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            return None

    if isinstance(content, list):
        for block in content:

            if not isinstance(block, dict):
                continue

            if block.get("type") == "text":
                text = block.get("text", "")

                result = parse_tool_result(text)

                if result:
                    return result

            if "json" in block:
                result = block.get("json")

                if isinstance(result, dict):
                    return result

    return None


# ---------------------------------------------------------
# Check Draft Result
# ---------------------------------------------------------


def draft_was_created(message: ToolMessage) -> bool:
    """
    Return True only when gmail_create_draft
    actually reports success.
    """

    if getattr(message, "name", None) != "gmail_create_draft":
        return False

    result = parse_tool_result(message.content)

    if result:
        return result.get("success") is True

    content_text = str(message.content).lower()

    return '"success": true' in content_text or "'success': true" in content_text


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


async def main():

    # -----------------------------------------------------
    # Multi MCP configuration
    # -----------------------------------------------------

    mcp_config = {
        "mcpServers": {
            # ---------------------------------------------
            # Weather MCP
            # ---------------------------------------------
            "weather": {
                "command": "uv",
                "args": [
                    "run",
                    "mcp",
                    "run",
                    "src/multi_mcp_agent/server.py",
                ],
            },
            # ---------------------------------------------
            # Gmail MCP
            # ---------------------------------------------
            "gmail": {
                "command": "uv",
                "args": [
                    "run",
                    "mcp",
                    "run",
                    "src/multi_mcp_agent/gmail_server.py",
                ],
            },
            # ---------------------------------------------
            # Fetch MCP
            # ---------------------------------------------
            "fetch": {
                "command": "uvx",
                "args": [
                    "--with",
                    "mcp>=1.29.0,<2",
                    "mcp-server-fetch",
                ],
            },
            # ---------------------------------------------
            # Git MCP
            # ---------------------------------------------
            "git": {
                "command": "uvx",
                "args": [
                    "--with",
                    "mcp>=1.29.0,<2",
                    "mcp-server-git",
                    "--repository",
                    PROJECT_REPO,
                ],
            },
        }
    }

    # -----------------------------------------------------
    # Gemini model
    # -----------------------------------------------------

    model = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0,
    )

    # -----------------------------------------------------
    # Connect to MCP servers
    # -----------------------------------------------------

    async with MCPAdapter(mcp_config) as adapter:

        # Discover MCP tools
        tools = await adapter.list_tools()

        print("\nAvailable MCP tools:")

        for tool in tools:
            print(f"- {tool.name}")

        # Give all MCP tools to Gemini
        model_with_tools = model.bind_tools(tools)

        # -------------------------------------------------
        # Agent Node
        # -------------------------------------------------

        async def agent_node(state: AgentState):

            system_message = SystemMessage(content=SYSTEM_PROMPT)

            response = await model_with_tools.ainvoke(
                [
                    system_message,
                    *state["messages"],
                ]
            )

            return {"messages": [response]}

        # -------------------------------------------------
        # Tool Node
        # -------------------------------------------------

        tool_node = ToolNode(tools)

        # -------------------------------------------------
        # Build LangGraph
        # -------------------------------------------------

        graph_builder = StateGraph(AgentState)

        graph_builder.add_node(
            "agent",
            agent_node,
        )

        graph_builder.add_node(
            "tools",
            tool_node,
        )

        # START -> Agent
        graph_builder.add_edge(
            START,
            "agent",
        )

        # Agent decision:
        #
        # tool requested -> tools node
        # no tool        -> END
        graph_builder.add_conditional_edges(
            "agent",
            tools_condition,
        )

        # Tool result -> Agent
        graph_builder.add_edge(
            "tools",
            "agent",
        )

        # Compile LangGraph
        graph = graph_builder.compile()

        # -------------------------------------------------
        # Conversation
        # -------------------------------------------------

        conversation_history = []

        print("\nMulti MCP Agent")
        print("Type 'exit' or 'quit' to stop.\n")

        while True:

            question = input("You: ").strip()

            if question.lower() in {
                "exit",
                "quit",
            }:
                print("Session ended.")
                break

            if not question:
                continue

            # -------------------------------------------------
            # Add user message
            # -------------------------------------------------

            conversation_history.append(HumanMessage(content=question))

            # Remember where this turn begins
            history_length_before_run = len(conversation_history)

            # -------------------------------------------------
            # Run LangGraph
            # -------------------------------------------------

            try:
                result = await graph.ainvoke({"messages": conversation_history})

            except GoogleRateLimitError:
                print(
                    "\nAI Error: Gemini API quota " "or rate limit has been reached.\n"
                )
                continue

            # -------------------------------------------------
            # Get only messages created this turn
            # -------------------------------------------------

            new_messages = result["messages"][history_length_before_run:]

            # Save complete conversation history
            conversation_history = result["messages"]

            # -------------------------------------------------
            # Print tool calls from this turn only
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

                        print(f"  Arguments: " f"{tool_call['args']}")

            # -------------------------------------------------
            # Detect Gmail draft success
            # -------------------------------------------------

            notification_sent = False

            for message in new_messages:

                if not isinstance(
                    message,
                    ToolMessage,
                ):
                    continue

                if draft_was_created(message) and not notification_sent:
                    notify_draft_created()

                    notification_sent = True

                    print("\nNotification: " "Gmail draft created successfully.")

            # -------------------------------------------------
            # Final AI response
            # -------------------------------------------------

            final_message = result["messages"][-1]

            print("\nAI:")

            print_message_content(final_message.content)

            print()


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
