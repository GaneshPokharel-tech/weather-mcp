import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # Start the MCP server through stdio
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "mcp", "run", "src/weather_mcp/server.py"],
    )

    # Connect to the MCP server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # -------------------------------------------------
            # Discover available MCP tools
            # -------------------------------------------------

            tools = await session.list_tools()

            for tool in tools.tools:
                print(tool.name)

            # Get city from user
            city = input("Enter city: ")

            # -------------------------------------------------
            # Call current weather tool
            # -------------------------------------------------

            current_result = await session.call_tool(
                "get_current_weather",
                arguments={
                    "city": city,
                },
            )

            if current_result.content:
                weather_data = json.loads(current_result.content[0].text)

                if "error" in weather_data:
                    print(f"Error: {weather_data['error']}")
                else:
                    print("Weather Data:", weather_data)

            # -------------------------------------------------
            # Call forecast tool
            # -------------------------------------------------

            forecast_result = await session.call_tool(
                "get_weather_forecast",
                arguments={
                    "city": city,
                    "days": 3,
                },
            )

            if forecast_result.content:
                forecast_data = json.loads(forecast_result.content[0].text)

                if "error" in forecast_data:
                    print(f"Forecast Error: {forecast_data['error']}")
                else:
                    print("Forecast Data:", forecast_data)

            # -------------------------------------------------
            # Read MCP resource
            # -------------------------------------------------

            about_resource = await session.read_resource("weather://about")

            print(
                "About Resource:",
                about_resource.contents[0].text,
            )

            # -------------------------------------------------
            # Get MCP prompt
            # -------------------------------------------------

            prompt_result = await session.get_prompt(
                "weather_report",
                arguments={
                    "city": city,
                },
            )

            print(
                "Prompt:",
                prompt_result.messages[0].content.text,
            )


if __name__ == "__main__":
    asyncio.run(main())
