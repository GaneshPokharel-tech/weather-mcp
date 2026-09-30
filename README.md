# Weather MCP

A learning project built to understand how the Model Context Protocol (MCP) works in a real application.

This project exposes weather functionality through an MCP server, connects to WeatherAPI.com, provides MCP tools, resources, and prompts, includes a custom Python MCP client, and integrates the MCP server with LangChain and Google Gemini.

---

## Project Goal

The main goal of this project is to learn the complete MCP flow:

```text
User
  ↓
LLM / MCP Client
  ↓
MCP Server
  ↓
MCP Tool
  ↓
WeatherAPI
  ↓
MCP Server
  ↓
LLM / Client
  ↓
Final Result
```

---

## Architecture

```text
                     User
                       │
                       ▼
               Google Gemini
                       │
                       ▼
                LangChain Agent
                       │
                       ▼
                  MCP Adapter
                       │
                       ▼
              Weather MCP Server
                 /      |      \
                /       |       \
             Tools   Resource   Prompt
               │
               ▼
         WeatherAPI.com
```

The project can also run without Gemini:

```text
Python MCP Client
       │
       ▼
Weather MCP Server
       │
       ▼
WeatherAPI.com
```

---

## Technologies Used

- Python 3.12
- MCP Python SDK
- LangChain
- Google Gemini
- WeatherAPI.com
- HTTPX
- python-dotenv
- uv
- MCP Inspector

---

## Project Structure

```text
weather-mcp/
│
├── .env
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock
│
└── src/
    └── weather_mcp/
        ├── __init__.py
        ├── server.py
        ├── client.py
        └── langchain_client.py
```

---

## Main Files

### `server.py`

Contains the MCP server.

It provides:

- current weather tool
- weather forecast tool
- location resolution
- MCP resource
- MCP prompt
- WeatherAPI integration
- input validation
- API error handling
- ambiguous-location handling

---

### `client.py`

A custom Python MCP client.

It demonstrates how to:

- connect to an MCP server using stdio
- initialize an MCP session
- discover MCP tools
- call MCP tools
- read MCP resources
- retrieve MCP prompts

---

### `langchain_client.py`

Connects the MCP server to LangChain and Google Gemini.

It allows Gemini to:

- discover MCP weather tools
- decide which tool to use
- call MCP tools through LangChain
- use weather data returned by MCP
- generate natural-language responses
- maintain conversation history during a session

---

# MCP Tools

## 1. `get_current_weather`

Gets the current weather for a location.

Example input:

```text
Kathmandu, Nepal
```

Example output:

```json
{
  "city": "Kathmandu",
  "country": "Nepal",
  "temperature_c": 18,
  "condition": "Patchy rain nearby",
  "humidity": 95,
  "wind_kph": 3.6,
  "feels_like_c": 20.1
}
```

---

## 2. `get_weather_forecast`

Gets a short-term weather forecast.

The project currently supports between 1 and 3 forecast days.

Example input:

```text
location: Palpa, Nepal
days: 2
```

Example output:

```json
{
  "city": "Tansen",
  "country": "Nepal",
  "forecast": [
    {
      "date": "2026-09-30",
      "max_temp_c": 30.3,
      "min_temp_c": 23.6,
      "condition": "Smoky haze",
      "chance_of_rain": 19
    },
    {
      "date": "2026-10-01",
      "max_temp_c": 30.5,
      "min_temp_c": 23.9,
      "condition": "Patchy rain nearby",
      "chance_of_rain": 57
    }
  ]
}
```

---

# Location Resolution

Weather locations can be ambiguous.

For example:

```text
Palpa
```

may match different places in different countries.

The MCP server first checks WeatherAPI's location search endpoint.

If multiple locations are found, the server does not guess.

Example:

```json
{
  "error": "Location is ambiguous",
  "message": "Please provide a more specific location, for example: 'Palpa, Nepal'."
}
```

The user can then provide:

```text
Palpa, Nepal
```

The server resolves the location and uses exact coordinates for the weather request.

This prevents weather information from being returned for the wrong country.

---

# MCP Resource

The project exposes this resource:

```text
weather://about
```

Example resource content:

```text
Weather MCP provides current weather and short-term weather forecasts using WeatherAPI.com.
```

A resource provides readable data or context to an MCP client.

---

# MCP Prompt

The server exposes this prompt:

```text
weather_report
```

The prompt accepts a location and creates a reusable weather-report instruction.

Example:

```text
weather_report("Kathmandu, Nepal")
```

A prompt provides reusable instructions for an AI model.

---

# Environment Variables

Create a `.env` file in the project root.

```env
WEATHER_API_KEY=your_weatherapi_key
GEMINI_API_KEY=your_gemini_api_key
```

Never commit `.env`.

Your `.gitignore` should contain:

```gitignore
.env
.venv
```

---

# Installation

## 1. Create virtual environment

```bash
uv venv
```

Activate it:

```bash
source .venv/bin/activate
```

---

## 2. Install MCP SDK

```bash
uv add "mcp[cli]"
```

---

## 3. Install HTTP dependencies

```bash
uv add httpx python-dotenv
```

---

## 4. Install LangChain MCP support

```bash
uv add "langchain[mcp]>=1.4.0"
```

---

## 5. Install Gemini integration

```bash
uv add langchain-google-genai
```

---

# Test with MCP Inspector

Run:

```bash
uv run mcp dev src/weather_mcp/server.py
```

Then open MCP Inspector.

The server should expose:

```text
Tools
├── get_current_weather
└── get_weather_forecast

Resource
└── weather://about

Prompt
└── weather_report
```

---

# Run the Python MCP Client

Run:

```bash
uv run python src/weather_mcp/client.py
```

Example:

```text
get_current_weather
get_weather_forecast

Enter city: Kathmandu

Weather Data: {...}

Forecast Data: {...}

About Resource:
Weather MCP provides current weather and short-term weather forecasts using WeatherAPI.com.

Prompt:
Provide a concise weather report for Kathmandu...
```

---

# Run the LangChain + Gemini Agent

Run:

```bash
uv run python src/weather_mcp/langchain_client.py
```

Example:

```text
You: Will it rain in Palpa tomorrow?

AI:
Could you please specify which Palpa you mean?
```

Then:

```text
You: Palpa, Nepal
```

The agent can use the previous conversation context and call:

```text
get_weather_forecast
```

with the correct location.

---

# Agent Flow

```text
User Question
      ↓
Google Gemini
      ↓
Tool Decision
      ↓
LangChain Agent
      ↓
MCP Adapter
      ↓
Weather MCP Server
      ↓
WeatherAPI
      ↓
MCP Tool Result
      ↓
Google Gemini
      ↓
Final Answer
```

---

# Error Handling

## Empty location

```json
{
  "error": "Location is required"
}
```

---

## Invalid location

```json
{
  "error": "No matching location found"
}
```

---

## Ambiguous location

```json
{
  "error": "Location is ambiguous"
}
```

---

## Invalid forecast days

```json
{
  "error": "Days must be between 1 and 3"
}
```

---

## Missing WeatherAPI key

```json
{
  "error": "WEATHER_API_KEY is missing"
}
```

---

## WeatherAPI connection failure

```json
{
  "error": "Unable to connect to WeatherAPI"
}
```

---

# Security

API keys are stored in `.env`.

They are not hard-coded in source files.

HTTPX logging is reduced so request URLs containing the WeatherAPI key are not printed in the terminal.

```python
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
```

---

# MCP Concepts Learned

## Tool

A tool performs an operation.

Examples:

```text
get_current_weather
get_weather_forecast
```

---

## Resource

A resource exposes readable context.

Example:

```text
weather://about
```

---

## Prompt

A prompt provides a reusable instruction template.

Example:

```text
weather_report
```

---

## Client

An MCP client connects to an MCP server and uses its capabilities.

In this project:

```text
client.py
```

is a custom Python MCP client.

---

## Server

An MCP server exposes:

```text
Tools
Resources
Prompts
```

In this project:

```text
server.py
```

is the MCP server.

---

## Stdio Transport

The client and server communicate locally using standard input and standard output.

```text
Client
  ↓
stdio
  ↓
MCP Server
```

---

# What This Project Demonstrates

Without MCP:

```text
Application
   ↓
WeatherAPI
```

With MCP:

```text
Application / AI
       ↓
      MCP
       ↓
Weather MCP Server
       ↓
WeatherAPI
```

With LangChain and Gemini:

```text
User
 ↓
Gemini
 ↓
LangChain Agent
 ↓
MCP
 ↓
WeatherAPI
 ↓
MCP
 ↓
Gemini
 ↓
User
```

The LLM does not directly access WeatherAPI.

The MCP server acts as the controlled interface between the AI application and the external service.

---

# Current Project Status

# Current Project Status

## Completed

- MCP server using Python
- WeatherAPI integration
- Current weather tool
- Weather forecast tool
- Location resolution
- Ambiguous location handling
- Input validation
- API error handling
- Environment variable configuration
- MCP Resource
- MCP Prompt
- MCP Inspector testing
- Custom Python MCP client
- Stdio transport
- LangChain MCP integration
- Google Gemini integration
- Automatic MCP tool selection
- Multi-turn conversation support
- Conversation history
- Gemini API rate-limit handling
- API-key log protection

The project is complete for the current learning scope.
---

## Remaining

- verify the complete multi-turn flow after Gemini quota becomes available
- handle Gemini `429 RESOURCE_EXHAUSTED` gracefully
- final code cleanup
- final end-to-end testing

---

# Learning Outcome

This project demonstrates how MCP can provide a structured and controlled connection between an AI application and an external API.

The final architecture is:

```text
User
 ↓
Gemini
 ↓
LangChain Agent
 ↓
MCP Adapter
 ↓
Weather MCP Server
 ↓
WeatherAPI
```

The project also demonstrates the three main MCP primitives:

```text
Tool
Resource
Prompt
```

and how an MCP client can discover and use them.