import logging
import os

import httpx
from dotenv import load_dotenv
from mcp.server import MCPServer

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

# Prevent HTTP request logs from exposing the API key
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

# Load environment variables from .env
load_dotenv()

# Read WeatherAPI key
API_KEY = os.getenv("WEATHER_API_KEY")


# ---------------------------------------------------------
# MCP Server
# ---------------------------------------------------------

mcp = MCPServer("Weather MCP")


# ---------------------------------------------------------
# Helper: Resolve location
# ---------------------------------------------------------


def resolve_location(location: str) -> dict:
    """
    Resolve a location before requesting weather data.

    This prevents WeatherAPI from silently choosing the wrong
    place when multiple locations have the same name.
    """

    url = "https://api.weatherapi.com/v1/search.json"

    params = {
        "key": API_KEY,
        "q": location,
    }

    try:
        response = httpx.get(
            url,
            params=params,
            timeout=10.0,
        )

        response.raise_for_status()

    except httpx.HTTPStatusError as exc:
        try:
            error_data = exc.response.json()

            message = error_data.get(
                "error",
                {},
            ).get(
                "message",
                "Location lookup failed",
            )

        except ValueError:
            message = "Location lookup failed"

        return {
            "error": message,
            "status_code": exc.response.status_code,
        }

    except httpx.RequestError:
        return {"error": "Unable to connect to WeatherAPI"}

    matches = response.json()

    # No location found
    if not matches:
        return {"error": "No matching location found"}

    # More than one result means the location may be ambiguous
    if len(matches) > 1:
        return {
            "error": "Location is ambiguous",
            "message": (
                "Please provide a more specific location, "
                "for example: 'Palpa, Nepal'."
            ),
            "matches": [
                {
                    "name": item["name"],
                    "region": item["region"],
                    "country": item["country"],
                }
                for item in matches[:5]
            ],
        }

    # Exactly one location matched
    match = matches[0]

    return {
        "name": match["name"],
        "region": match["region"],
        "country": match["country"],
        "lat": match["lat"],
        "lon": match["lon"],
    }


# ---------------------------------------------------------
# TOOL 1: Get current weather
# ---------------------------------------------------------


@mcp.tool()
def get_current_weather(location: str) -> dict:
    """
    Get current weather for a location.

    Use city and country when the place may be ambiguous,
    for example: "Palpa, Nepal".
    """

    # Validate location input
    if not location.strip():
        return {"error": "Location is required"}

    # Make sure API key exists
    if not API_KEY:
        return {"error": "WEATHER_API_KEY is missing"}

    # Resolve the location first
    resolved = resolve_location(location)

    # Stop if location resolution failed
    if "error" in resolved:
        return resolved

    # Use exact coordinates returned by location search
    coordinates = f"{resolved['lat']},{resolved['lon']}"

    url = "https://api.weatherapi.com/v1/current.json"

    params = {
        "key": API_KEY,
        "q": coordinates,
        "aqi": "no",
    }

    try:
        response = httpx.get(
            url,
            params=params,
            timeout=10.0,
        )

        response.raise_for_status()

    except httpx.HTTPStatusError as exc:
        try:
            error_data = exc.response.json()

            message = error_data.get(
                "error",
                {},
            ).get(
                "message",
                "Weather API request failed",
            )

        except ValueError:
            message = "Weather API request failed"

        return {
            "error": message,
            "status_code": exc.response.status_code,
        }

    except httpx.RequestError:
        return {"error": "Unable to connect to WeatherAPI"}

    data = response.json()

    return {
        "city": data["location"]["name"],
        "region": data["location"]["region"],
        "country": data["location"]["country"],
        "temperature_c": data["current"]["temp_c"],
        "condition": data["current"]["condition"]["text"],
        "humidity": data["current"]["humidity"],
        "wind_kph": data["current"]["wind_kph"],
        "feels_like_c": data["current"]["feelslike_c"],
    }


# ---------------------------------------------------------
# TOOL 2: Get weather forecast
# ---------------------------------------------------------


@mcp.tool()
def get_weather_forecast(
    location: str,
    days: int = 3,
) -> dict:
    """
    Get a weather forecast for a location.

    Use city and country when the place may be ambiguous,
    for example: "Palpa, Nepal".
    """

    # Validate location input
    if not location.strip():
        return {"error": "Location is required"}

    # Limit forecast to 1-3 days
    if days < 1 or days > 3:
        return {"error": "Days must be between 1 and 3"}

    # Make sure API key exists
    if not API_KEY:
        return {"error": "WEATHER_API_KEY is missing"}

    # Resolve location first
    resolved = resolve_location(location)

    # Stop if location resolution failed
    if "error" in resolved:
        return resolved

    # Use exact coordinates
    coordinates = f"{resolved['lat']},{resolved['lon']}"

    url = "https://api.weatherapi.com/v1/forecast.json"

    params = {
        "key": API_KEY,
        "q": coordinates,
        "days": days,
        "aqi": "no",
        "alerts": "no",
    }

    try:
        response = httpx.get(
            url,
            params=params,
            timeout=10.0,
        )

        response.raise_for_status()

    except httpx.HTTPStatusError as exc:
        try:
            error_data = exc.response.json()

            message = error_data.get(
                "error",
                {},
            ).get(
                "message",
                "Weather API request failed",
            )

        except ValueError:
            message = "Weather API request failed"

        return {
            "error": message,
            "status_code": exc.response.status_code,
        }

    except httpx.RequestError:
        return {"error": "Unable to connect to WeatherAPI"}

    data = response.json()

    forecast = []

    for day in data["forecast"]["forecastday"]:
        forecast.append(
            {
                "date": day["date"],
                "max_temp_c": day["day"]["maxtemp_c"],
                "min_temp_c": day["day"]["mintemp_c"],
                "condition": day["day"]["condition"]["text"],
                "chance_of_rain": (day["day"]["daily_chance_of_rain"]),
            }
        )

    return {
        "city": data["location"]["name"],
        "region": data["location"]["region"],
        "country": data["location"]["country"],
        "forecast": forecast,
    }


# ---------------------------------------------------------
# RESOURCE
# ---------------------------------------------------------


@mcp.resource("weather://about")
def get_about() -> str:
    """Describe what the Weather MCP server can do."""

    return (
        "Weather MCP provides current weather and short-term "
        "weather forecasts using WeatherAPI.com. "
        "Locations are resolved before weather requests to reduce "
        "incorrect matches for places with the same name."
    )


# ---------------------------------------------------------
# PROMPT
# ---------------------------------------------------------


@mcp.prompt()
def weather_report(location: str) -> str:
    """Create a reusable prompt for a weather report."""

    return (
        f"Provide a concise weather report for {location}. "
        "Use the available weather tools to include current conditions "
        "and the short-term forecast. "
        "If the location is ambiguous, use the city and country "
        "instead of guessing."
    )
