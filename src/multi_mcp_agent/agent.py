import asyncio

import json

import queue

import subprocess

import threading

from pathlib import Path

from typing import Annotated

import os

from datetime import datetime

from zoneinfo import ZoneInfo


from dotenv import load_dotenv

from langchain.mcp import MCPAdapter

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_google_genai.chat_models import GoogleRateLimitError

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langgraph.graph import END, START, StateGraph

from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode

from langgraph.types import Command, interrupt

from typing_extensions import TypedDict

# =========================================================

# Environment

# =========================================================


load_dotenv()


# =========================================================

# Paths

# =========================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


MCP_SERVER_PATH = PROJECT_ROOT / "src" / "multi_mcp_agent" / "mcp_server.py"


CHECKPOINT_DB = PROJECT_ROOT / ".langgraph_checkpoints.sqlite"


# =========================================================

# Runtime Settings

# =========================================================


MODEL_NAME = "gemini-3.5-flash-lite"


MODEL_TIMEOUT_SECONDS = 35

TURN_TIMEOUT_SECONDS = 90


MODEL_MAX_ATTEMPTS = 3

RETRY_BASE_DELAY = 0.75


DEFAULT_TIMEZONE = os.getenv(
    "CALENDAR_TIMEZONE",
    "Asia/Kathmandu",
)


# =========================================================

# Tools Requiring Human Approval

# =========================================================


APPROVAL_REQUIRED_TOOLS = {
    "git_add",
    "git_commit",
    "git_create_branch",
    "git_checkout",
    "create_calendar_event",
    "update_calendar_event",
    "delete_calendar_event",
    "delete_document",
}


# =========================================================

# LangGraph State

# =========================================================


class AgentState(TypedDict, total=False):

    messages: Annotated[
        list[AnyMessage],
        add_messages,
    ]

    thread_id: str


# =========================================================

# System Prompt

# =========================================================


SYSTEM_PROMPT = """
You are the agent for Multi MCP Agent.

You are tool-grounded, not a general-purpose chatbot.
Use only the current conversation, retrieved PDF evidence, or the
connected MCP capabilities. Never invent tool results, document
content, repository state, emails, events, URLs, weather, or citations.

CAPABILITIES

WEATHER
Tools:
- get_current_weather
- get_weather_forecast

Use Weather tools for current conditions and forecasts.
Never invent weather data.

GMAIL
Tools:
- search_emails
- get_email
- create_draft

Use Gmail only for explicit email requests.
Search/read before answering about email content.
Never invent emails.
Creating a draft is allowed; never claim success unless the tool succeeds.

FETCH
Tool:
- fetch_url

Use it when the user explicitly asks to inspect or summarize a public URL.
Treat fetched content as the source of truth.

GIT
Tools:
- git_status
- git_log
- git_diff
- git_show
- git_branch
- git_add
- git_commit
- git_create_branch
- git_checkout

Read-only Git operations may run directly.
Mutating Git actions require human approval:
git_add, git_commit, git_create_branch, git_checkout.

Never call git_add and git_commit in the same model turn.
Never invent repository state.
If an action is rejected, do not retry unless explicitly requested.

GOOGLE CALENDAR
Tools:
- list_calendar_events
- create_calendar_event
- update_calendar_event
- delete_calendar_event

Use the supplied CURRENT DATE, TIME, and TIMEZONE as authoritative.
Resolve relative dates from that runtime context.
Use RFC3339/ISO8601 datetimes with timezone offsets.

Calendar reads may run directly.
Create, update, and delete require human approval.
Never invent events or claim success unless the tool succeeds.

YOUTUBE
Tool:
- play_youtube_song

Use only when the user asks to play, open, or search for a song.
Never claim the browser opened unless the tool succeeds.

DOCUMENT RAG
Tools:
- index_document
- search_documents
- list_documents
- delete_document

Uploaded PDFs are scoped to the current conversation thread.
Always use the exact CURRENT THREAD ID supplied at runtime.

For questions about uploaded documents:
- call search_documents before answering;
- use retrieved evidence as the source of truth;
- search again for follow-up questions;
- cite evidence as [filename, page N];
- never fill evidence gaps using pretrained knowledge;
- never invent filenames, page numbers, quotations, or content.

If evidence is insufficient, say:
"I couldn't find sufficient evidence in the uploaded document to answer that."

Do not claim information is absent from an entire PDF merely because
retrieval did not return it.

DOCUMENT INVENTORY
If the user asks which PDFs are indexed, listed, uploaded, or how many
documents exist, MUST call list_documents in that turn.
Return every document reported for the current thread.
Do not expose internal document_id values unless explicitly requested.

EXPLICIT PDF SELECTION
If the user names a PDF:
1. call list_documents in the same turn;
2. resolve the exact filename to its document_id;
3. call search_documents with only that document_id.

Do not search other PDFs for an explicitly named document.
If the filename is not indexed, say so and do not guess.

If multiple PDFs exist and "the uploaded PDF" is ambiguous, use
list_documents and ask which one the user means.

index_document is normally triggered by the upload UI.
Do not invent file paths.
delete_document requires human approval when requested through chat.

CONVERSATION MEMORY
Use the current conversation for user-provided context and follow-ups.
Do not search Gmail or unrelated tools merely to answer conversational
memory questions.
If the information is not present, do not guess.

CAPABILITY BOUNDARY
Allowed sources are:
- current conversation;
- uploaded PDF evidence;
- Weather;
- Gmail;
- Google Calendar;
- Fetch;
- Git;
- YouTube.

Do not answer unrelated general-knowledge questions from pretrained
knowledge.
Do not generate fictional stories, poems, invented scenarios, or
unsupported creative content.

If a request is outside these capabilities, respond briefly:
"That request is outside the connected capabilities of this agent."

GENERAL
Use only tools relevant to the request.
Do not call unrelated tools.
When multiple tools are needed, use them in the correct order.
Use tool results as authoritative.
Keep answers concise unless the user asks for detail.
"""



# =========================================================

# Message Helpers

# =========================================================


def message_to_text(content) -> str:

    if isinstance(content, str):

        return content

    if isinstance(content, list):

        parts = []

        for block in content:

            if not isinstance(block, dict):

                continue

            text = block.get("text")

            if isinstance(text, str) and text:

                parts.append(text)

        return "\n".join(parts)

    if content is None:

        return ""

    return str(content)


# =========================================================

# Tool Result Helpers

# =========================================================


def parse_tool_result(content):

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

            if isinstance(
                block.get("json"),
                dict,
            ):

                return block["json"]

            if isinstance(
                block.get("text"),
                str,
            ):

                result = parse_tool_result(block["text"])

                if result:

                    return result

    return None


def draft_was_created(
    message: ToolMessage,
) -> bool:

    name = getattr(message, "name", "") or ""

    if not name.endswith("create_draft"):

        return False

    result = parse_tool_result(message.content)

    if result:

        return result.get("success") is True

    text = str(message.content).lower()

    return '"success": true' in text or "'success': true" in text


# =========================================================

# Notification

# =========================================================


def notify_draft_created():

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


# =========================================================

# Error Handling

# =========================================================


def is_transient_error(
    error: Exception,
) -> bool:

    text = str(error).lower()

    markers = [
        "503",
        "502",
        "504",
        "unavailable",
        "high demand",
        "temporarily unavailable",
        "deadline exceeded",
        "connection reset",
        "connection aborted",
        "connection error",
        "timeout",
        "timed out",
    ]

    return any(marker in text for marker in markers)


def classify_error(
    error: Exception,
) -> dict:

    if isinstance(
        error,
        GoogleRateLimitError,
    ):

        return {
            "kind": "quota",
            "message": ("Gemini API quota or rate limit " "has been reached."),
        }

    if isinstance(
        error,
        (
            asyncio.TimeoutError,
            TimeoutError,
        ),
    ):

        return {
            "kind": "timeout",
            "message": ("The request took too long " "and timed out."),
        }

    text = str(error)

    lowered = text.lower()

    if "429" in lowered or "resource_exhausted" in lowered or "quota" in lowered:

        return {
            "kind": "quota",
            "message": ("Gemini API quota or rate limit " "has been reached."),
        }

    if "503" in lowered or "unavailable" in lowered or "high demand" in lowered:

        return {
            "kind": "unavailable",
            "message": (
                "Gemini is temporarily unavailable. " "Please try again shortly."
            ),
        }

    return {
        "kind": "error",
        "message": text,
    }


# =========================================================

# MCP Configuration

# =========================================================


def get_mcp_config():

    return {
        "mcpServers": {
            "multi": {
                "command": "uv",
                "args": [
                    "run",
                    "mcp",
                    "run",
                    str(MCP_SERVER_PATH),
                ],
            }
        }
    }


# =========================================================

# Model Retry

# =========================================================


async def invoke_model_with_retry(
    model_with_tools,
    messages,
):

    last_error = None

    for attempt in range(MODEL_MAX_ATTEMPTS):

        try:

            return await asyncio.wait_for(
                model_with_tools.ainvoke(messages),
                timeout=MODEL_TIMEOUT_SECONDS,
            )

        except GoogleRateLimitError:

            raise

        except asyncio.TimeoutError as exc:

            last_error = exc

        except Exception as exc:

            if not is_transient_error(exc):

                raise

            last_error = exc

        if attempt < MODEL_MAX_ATTEMPTS - 1:

            delay = RETRY_BASE_DELAY * (2**attempt)

            await asyncio.sleep(delay)

    if last_error:

        raise last_error

    raise RuntimeError("Model request failed.")


# =========================================================

# Graph Helpers

# =========================================================


def get_tool_calls(
    state: AgentState,
) -> list[dict]:

    if not state["messages"]:

        return []

    last_message = state["messages"][-1]

    return (
        getattr(
            last_message,
            "tool_calls",
            None,
        )
        or []
    )


def route_after_agent(
    state: AgentState,
):
    """

    Decide whether to:

    - finish

    - execute safe tools

    - ask human approval

    """

    tool_calls = get_tool_calls(state)

    if not tool_calls:

        return END

    requires_approval = any(
        tool_call.get("name") in APPROVAL_REQUIRED_TOOLS for tool_call in tool_calls
    )

    if requires_approval:

        return "approval"

    return "tools"


# =========================================================

# Build LangGraph

# =========================================================


def build_graph(
    tools,
    model_with_tools,
    checkpointer,
):

    # -----------------------------------------------------

    # Agent Node

    # -----------------------------------------------------

    async def agent_node(
        state: AgentState,
    ):
        current_datetime = datetime.now(ZoneInfo(DEFAULT_TIMEZONE))

        current_thread_id = state.get(
            "thread_id",
            "",
        )

        current_context = f"""

CURRENT DATE AND TIME

Current datetime: {current_datetime.isoformat()}
Current date: {current_datetime.date().isoformat()}
Current timezone: {DEFAULT_TIMEZONE}

IMPORTANT CALENDAR RULES:
- The date and time above are authoritative.
- Resolve "today" from Current date.
- Resolve "tomorrow" as exactly one calendar day after Current date.
- Resolve "yesterday" as exactly one calendar day before Current date.
- Resolve "next Monday", "next week", and similar expressions
  relative to Current date.
- Never use an internal or remembered model date for calendar operations.
- Before creating or updating a Calendar event, verify that the
  generated date matches the user's relative-date request.

CURRENT CONVERSATION THREAD

Current thread ID: {current_thread_id}

IMPORTANT RAG RULE:
- Use exactly the Current thread ID above as the thread_id argument
  for list_documents, search_documents, and delete_document.
"""

        messages = [
            SystemMessage(content=(SYSTEM_PROMPT + current_context)),
            *state["messages"],
        ]

        response = await invoke_model_with_retry(
            model_with_tools,
            messages,
        )

        return {"messages": [response]}

    # -----------------------------------------------------

    # Human Approval Node

    # -----------------------------------------------------

    def approval_node(
        state: AgentState,
    ):

        tool_calls = get_tool_calls(state)

        protected_calls = [
            {
                "name": tool_call.get("name"),
                "args": tool_call.get(
                    "args",
                    {},
                ),
                "id": tool_call.get("id"),
            }
            for tool_call in tool_calls
            if (tool_call.get("name") in APPROVAL_REQUIRED_TOOLS)
        ]

        decision = interrupt(
            {
                "type": "tool_approval",
                "title": ("Human approval required"),
                "message": (
                    "The agent wants to perform "
                    "a protected action that can modify data. "
                    "Review the tool and arguments carefully "
                    "before approving."
                ),
                "tool_calls": protected_calls,
            }
        )

        action = ""

        if isinstance(
            decision,
            dict,
        ):

            action = decision.get(
                "action",
                "",
            ).lower()

        # ---------------------------------------------

        # Approved

        # ---------------------------------------------

        if action == "approve":

            return Command(goto="tools")

        # ---------------------------------------------

        # Rejected

        # ---------------------------------------------

        denied_messages = []

        for tool_call in tool_calls:

            denied_messages.append(
                ToolMessage(
                    content=(
                        "The user rejected this "
                        "tool action. Do not execute "
                        "or retry it unless the user "
                        "explicitly asks again."
                    ),
                    tool_call_id=(tool_call.get("id")),
                    name=(tool_call.get("name")),
                )
            )

        return Command(
            goto="agent",
            update={"messages": (denied_messages)},
        )

    # -----------------------------------------------------

    # Tool Node

    # -----------------------------------------------------

    tool_node = ToolNode(tools)

    # -----------------------------------------------------

    # Graph

    # -----------------------------------------------------

    builder = StateGraph(AgentState)

    builder.add_node(
        "agent",
        agent_node,
    )

    builder.add_node(
        "approval",
        approval_node,
    )

    builder.add_node(
        "tools",
        tool_node,
    )

    builder.add_edge(
        START,
        "agent",
    )

    builder.add_conditional_edges(
        "agent",
        route_after_agent,
    )

    builder.add_edge(
        "tools",
        "agent",
    )

    return builder.compile(checkpointer=checkpointer)


# =========================================================

# Agent Runtime

# =========================================================


class AgentRuntime:

    def __init__(self):

        self.loop = asyncio.new_event_loop()

        self.thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
        )

        self.thread.start()

        future = asyncio.run_coroutine_threadsafe(
            self._initialize(),
            self.loop,
        )

        future.result()

    # -----------------------------------------------------

    # Event Loop

    # -----------------------------------------------------

    def _run_loop(self):

        asyncio.set_event_loop(self.loop)

        self.loop.run_forever()

    # -----------------------------------------------------

    # Initialize

    # -----------------------------------------------------

    async def _initialize(self):

        # MCP

        self.adapter = MCPAdapter(get_mcp_config())

        await self.adapter.__aenter__()

        self.tools = await self.adapter.list_tools()

        # Gemini

        self.model = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=0,
        )

        self.model_with_tools = self.model.bind_tools(self.tools)

        # SQLite

        self.checkpointer_context = AsyncSqliteSaver.from_conn_string(
            str(CHECKPOINT_DB)
        )

        self.checkpointer = await self.checkpointer_context.__aenter__()

        # Graph

        self.graph = build_graph(
            self.tools,
            self.model_with_tools,
            self.checkpointer,
        )

    # -----------------------------------------------------

    # Tool Count

    # -----------------------------------------------------

    @property
    def tool_count(self):

        return len(self.tools)

    # -----------------------------------------------------

    # Direct MCP Tool Call

    # -----------------------------------------------------

    def call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ):

        future = asyncio.run_coroutine_threadsafe(
            self._call_tool(
                tool_name,
                arguments,
            ),
            self.loop,
        )

        return future.result()

    async def _call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ):

        selected_tool = None

        for tool in self.tools:

            name = getattr(
                tool,
                "name",
                "",
            )

            if name == tool_name or name.endswith(tool_name):

                selected_tool = tool

                break

        if selected_tool is None:

            raise RuntimeError(f"MCP tool not found: {tool_name}")

        result = await selected_tool.ainvoke(arguments)

        parsed = parse_tool_result(result)

        return parsed if parsed is not None else result

    # -----------------------------------------------------

    # Load Messages

    # -----------------------------------------------------

    def load_messages(
        self,
        thread_id: str,
    ):

        future = asyncio.run_coroutine_threadsafe(
            self._load_messages(thread_id),
            self.loop,
        )

        return future.result()

    async def _load_messages(
        self,
        thread_id: str,
    ):

        config = {
            "configurable": {
                "thread_id": (thread_id),
            }
        }

        checkpoint = await self.checkpointer.aget(config)

        if not checkpoint:

            return []

        return checkpoint.get(
            "channel_values",
            {},
        ).get(
            "messages",
            [],
        )

    # -----------------------------------------------------

    # Delete Thread

    # -----------------------------------------------------

    def delete_thread(
        self,
        thread_id: str,
    ):

        future = asyncio.run_coroutine_threadsafe(
            self.checkpointer.adelete_thread(thread_id),
            self.loop,
        )

        return future.result()

    # -----------------------------------------------------

    # Chat History

    # -----------------------------------------------------

    def list_threads(
        self,
        limit: int = 30,
    ):

        future = asyncio.run_coroutine_threadsafe(
            self._list_threads(limit),
            self.loop,
        )

        return future.result()

    async def _list_threads(
        self,
        limit: int,
    ):

        threads = []

        seen = set()

        async for item in self.checkpointer.alist(
            None,
            limit=500,
        ):

            config = item.config or {}

            thread_id = config.get(
                "configurable",
                {},
            ).get("thread_id")

            if not thread_id:

                continue

            if thread_id in seen:

                continue

            seen.add(thread_id)

            checkpoint = item.checkpoint or {}

            messages = checkpoint.get(
                "channel_values",
                {},
            ).get(
                "messages",
                [],
            )

            title = "New chat"

            for message in messages:

                if isinstance(
                    message,
                    HumanMessage,
                ):

                    text = message_to_text(message.content).strip()

                    if text:

                        title = text

                        break

            if len(title) > 42:

                title = title[:39] + "..."

            threads.append(
                {
                    "thread_id": (thread_id),
                    "title": title,
                    "updated_at": (
                        checkpoint.get(
                            "ts",
                            "",
                        )
                    ),
                }
            )

            if len(threads) >= limit:

                break

        return threads

    # -----------------------------------------------------

    # Pending Human Approval

    # -----------------------------------------------------

    def get_pending_approval(
        self,
        thread_id: str,
    ):

        future = asyncio.run_coroutine_threadsafe(
            self._get_pending_approval(thread_id),
            self.loop,
        )

        return future.result()

    async def _get_pending_approval(
        self,
        thread_id: str,
    ):

        config = {
            "configurable": {
                "thread_id": (thread_id),
            }
        }

        snapshot = await self.graph.aget_state(config)

        if not snapshot:

            return None

        tasks = (
            getattr(
                snapshot,
                "tasks",
                None,
            )
            or []
        )

        for task in tasks:

            interrupts = (
                getattr(
                    task,
                    "interrupts",
                    None,
                )
                or []
            )

            for pending in interrupts:

                value = getattr(
                    pending,
                    "value",
                    None,
                )

                if (
                    isinstance(
                        value,
                        dict,
                    )
                    and value.get("type") == "tool_approval"
                ):

                    return value

        return None

    # -----------------------------------------------------

    # Stream New User Turn

    # -----------------------------------------------------

    def stream_turn(
        self,
        user_message: str,
        thread_id: str,
    ):

        event_queue = queue.Queue()

        future = asyncio.run_coroutine_threadsafe(
            self._drive_graph(
                {
                    "messages": [HumanMessage(content=(user_message))],
                    "thread_id": thread_id,
                },
                thread_id,
                event_queue,
            ),
            self.loop,
        )

        yield from self._consume_queue(
            event_queue,
            future,
        )

    # -----------------------------------------------------

    # Resume Human Approval

    # -----------------------------------------------------

    def resume_approval(
        self,
        thread_id: str,
        approved: bool,
    ):

        event_queue = queue.Queue()

        action = "approve" if approved else "reject"

        future = asyncio.run_coroutine_threadsafe(
            self._drive_graph(
                Command(resume={"action": (action)}),
                thread_id,
                event_queue,
            ),
            self.loop,
        )

        yield from self._consume_queue(
            event_queue,
            future,
        )

    # -----------------------------------------------------

    # Queue Consumer

    # -----------------------------------------------------

    def _consume_queue(
        self,
        event_queue,
        future,
    ):

        while True:

            event = event_queue.get()

            yield event

            if event.get("type") in {
                "done",
                "error",
                "approval",
            }:

                break

        try:

            future.result()

        except Exception:

            # Error event is already sent

            # to Streamlit.

            pass

    # -----------------------------------------------------

    # Drive Graph

    # -----------------------------------------------------

    async def _drive_graph(
        self,
        graph_input,
        thread_id: str,
        event_queue,
    ):

        config = {
            "configurable": {
                "thread_id": (thread_id),
            }
        }

        try:

            # -----------------------------------------

            # Existing message count

            # -----------------------------------------

            checkpoint = await self.checkpointer.aget(config)

            previous_messages = []

            if checkpoint:

                previous_messages = checkpoint.get(
                    "channel_values",
                    {},
                ).get(
                    "messages",
                    [],
                )

            previous_count = len(previous_messages)

            # -----------------------------------------

            # Stream Graph

            # -----------------------------------------

            async with asyncio.timeout(TURN_TIMEOUT_SECONDS):

                async for (
                    message_chunk,
                    metadata,
                ) in self.graph.astream(
                    graph_input,
                    config=config,
                    stream_mode="messages",
                ):

                    if not isinstance(
                        message_chunk,
                        AIMessageChunk,
                    ):

                        continue

                    text = message_to_text(message_chunk.content)

                    if text:

                        event_queue.put(
                            {
                                "type": "token",
                                "text": text,
                            }
                        )

            # -----------------------------------------

            # Check whether LangGraph paused

            # for human approval

            # -----------------------------------------

            pending_approval = await self._get_pending_approval(thread_id)

            if pending_approval:

                event_queue.put(
                    {
                        "type": ("approval"),
                        "approval": (pending_approval),
                    }
                )

                return

            # -----------------------------------------

            # Completed State

            # -----------------------------------------

            completed_checkpoint = await self.checkpointer.aget(config)

            if not completed_checkpoint:

                raise RuntimeError("No LangGraph checkpoint " "was created.")

            all_messages = completed_checkpoint.get(
                "channel_values",
                {},
            ).get(
                "messages",
                [],
            )

            new_messages = all_messages[previous_count:]

            # -----------------------------------------

            # Tools Used

            # -----------------------------------------

            tools_used = []

            for message in new_messages:

                tool_calls = getattr(
                    message,
                    "tool_calls",
                    None,
                )

                if not tool_calls:

                    continue

                for tool_call in tool_calls:

                    name = tool_call.get("name")

                    if name and name not in tools_used:

                        tools_used.append(name)

            # -----------------------------------------

            # Gmail Draft Notification

            # -----------------------------------------

            draft_created = False

            for message in new_messages:

                if not isinstance(
                    message,
                    ToolMessage,
                ):

                    continue

                if draft_was_created(message):

                    draft_created = True

                    break

            if draft_created:

                notify_draft_created()

            # -----------------------------------------

            # Final Assistant Response

            # -----------------------------------------

            final_text = ""

            for message in reversed(all_messages):

                if not isinstance(
                    message,
                    AIMessage,
                ):

                    continue

                text = message_to_text(message.content).strip()

                if text:

                    final_text = text

                    break

            event_queue.put(
                {
                    "type": "done",
                    "response": (final_text),
                    "tools_used": (tools_used),
                    "draft_created": (draft_created),
                }
            )

        except Exception as exc:

            error = classify_error(exc)

            event_queue.put(
                {
                    "type": "error",
                    **error,
                }
            )
