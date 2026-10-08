import base64

import ipaddress

import logging

import os

import socket

import subprocess
import webbrowser
from datetime import datetime
from zoneinfo import ZoneInfo

from yt_dlp import YoutubeDL

from email.message import EmailMessage

from html.parser import HTMLParser

from pathlib import Path

from urllib.parse import urljoin, urlparse
import httpx
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from mcp.server import MCPServer
from multi_mcp_agent.rag import (
    delete_document as rag_delete_document,
    index_pdf as rag_index_pdf,
    list_documents as rag_list_documents,
    search_documents as rag_search_documents,
)

# =========================================================

# Environment

# =========================================================


load_dotenv()


logging.getLogger("httpx").setLevel(logging.WARNING)

logging.getLogger("httpcore").setLevel(logging.WARNING)


WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")


GMAIL_CLIENT_ID = os.getenv("GMAIL_OAUTH_CLIENT_ID")

GMAIL_CLIENT_SECRET = os.getenv("GMAIL_OAUTH_CLIENT_SECRET")


GMAIL_TOKEN_FILE = Path(".gmail_token.json")


GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]

CALENDAR_TOKEN_FILE = Path(".calendar_token.json")


CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
]


CALENDAR_TIMEZONE = os.getenv(
    "CALENDAR_TIMEZONE",
    "Asia/Kathmandu",
)


# Automatically resolves:

# /Users/ganesh/projects/multi-mcp-agent

PROJECT_REPO = Path(__file__).resolve().parents[2]


MAX_FETCH_CHARS = 20000


# =========================================================

# MCP Server

# =========================================================


mcp = MCPServer("Multi MCP Server")


# =========================================================

# WEATHER HELPERS

# =========================================================


def resolve_location(location: str) -> dict:

    if not WEATHER_API_KEY:

        return {
            "error": "WEATHER_API_KEY is missing",
        }

    try:

        response = httpx.get(
            "https://api.weatherapi.com/v1/search.json",
            params={
                "key": WEATHER_API_KEY,
                "q": location,
            },
            timeout=10,
        )

        response.raise_for_status()

        matches = response.json()

        if not matches:

            return {
                "error": "No matching location found",
            }

        if len(matches) > 1:

            return {
                "error": "Location is ambiguous",
                "message": (
                    "Please provide a more specific location, "
                    "for example: 'Palpa, Nepal'."
                ),
                "matches": [
                    {
                        "name": item.get("name"),
                        "region": item.get("region"),
                        "country": item.get("country"),
                        "lat": item.get("lat"),
                        "lon": item.get("lon"),
                    }
                    for item in matches[:5]
                ],
            }

        match = matches[0]

        return {
            "name": match.get("name"),
            "region": match.get("region"),
            "country": match.get("country"),
            "lat": match.get("lat"),
            "lon": match.get("lon"),
        }

    except httpx.HTTPStatusError as exc:

        try:

            error_data = exc.response.json()

            message = error_data.get("error", {}).get(
                "message",
                "WeatherAPI request failed",
            )

        except Exception:

            message = "WeatherAPI request failed"

        return {
            "error": message,
        }

    except httpx.RequestError:

        return {
            "error": "Unable to connect to WeatherAPI",
        }


# =========================================================

# WEATHER TOOLS

# =========================================================


@mcp.tool()
def get_current_weather(location: str) -> dict:
    """

    Get current weather for a location.

    """

    if not location.strip():

        return {
            "error": "Location is required",
        }

    if not WEATHER_API_KEY:

        return {
            "error": "WEATHER_API_KEY is missing",
        }

    resolved = resolve_location(location)

    if "error" in resolved:

        return resolved

    coordinates = f"{resolved['lat']},{resolved['lon']}"

    try:

        response = httpx.get(
            "https://api.weatherapi.com/v1/current.json",
            params={
                "key": WEATHER_API_KEY,
                "q": coordinates,
            },
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        location_data = data.get(
            "location",
            {},
        )

        current = data.get(
            "current",
            {},
        )

        return {
            "city": location_data.get("name"),
            "region": location_data.get("region"),
            "country": location_data.get("country"),
            "temperature_c": current.get("temp_c"),
            "condition": (current.get("condition", {}).get("text")),
            "humidity": current.get("humidity"),
            "wind_kph": current.get("wind_kph"),
            "feels_like_c": current.get("feelslike_c"),
        }

    except httpx.HTTPStatusError as exc:

        try:

            error_data = exc.response.json()

            message = error_data.get("error", {}).get(
                "message",
                "WeatherAPI request failed",
            )

        except Exception:

            message = "WeatherAPI request failed"

        return {
            "error": message,
        }

    except httpx.RequestError:

        return {
            "error": "Unable to connect to WeatherAPI",
        }


@mcp.tool()
def get_weather_forecast(
    location: str,
    days: int = 3,
) -> dict:
    """

    Get a 1-to-3-day weather forecast.

    """

    if not location.strip():

        return {
            "error": "Location is required",
        }

    if days < 1 or days > 3:

        return {
            "error": "Days must be between 1 and 3",
        }

    if not WEATHER_API_KEY:

        return {
            "error": "WEATHER_API_KEY is missing",
        }

    resolved = resolve_location(location)

    if "error" in resolved:

        return resolved

    coordinates = f"{resolved['lat']},{resolved['lon']}"

    try:

        response = httpx.get(
            "https://api.weatherapi.com/v1/forecast.json",
            params={
                "key": WEATHER_API_KEY,
                "q": coordinates,
                "days": days,
            },
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        location_data = data.get(
            "location",
            {},
        )

        forecast_days = []

        for item in data.get("forecast", {}).get("forecastday", []):

            day = item.get(
                "day",
                {},
            )

            forecast_days.append(
                {
                    "date": item.get("date"),
                    "max_temp_c": day.get("maxtemp_c"),
                    "min_temp_c": day.get("mintemp_c"),
                    "condition": (day.get("condition", {}).get("text")),
                    "chance_of_rain": (day.get("daily_chance_of_rain")),
                }
            )

        return {
            "city": location_data.get("name"),
            "region": location_data.get("region"),
            "country": location_data.get("country"),
            "forecast": forecast_days,
        }

    except httpx.HTTPStatusError as exc:

        try:

            error_data = exc.response.json()

            message = error_data.get("error", {}).get(
                "message",
                "WeatherAPI request failed",
            )

        except Exception:

            message = "WeatherAPI request failed"

        return {
            "error": message,
        }

    except httpx.RequestError:

        return {
            "error": "Unable to connect to WeatherAPI",
        }


# =========================================================

# GMAIL AUTHENTICATION

# =========================================================


def get_gmail_credentials() -> Credentials:

    if not GMAIL_CLIENT_ID or not GMAIL_CLIENT_SECRET:

        raise RuntimeError(
            "Missing GMAIL_OAUTH_CLIENT_ID or " "GMAIL_OAUTH_CLIENT_SECRET in .env"
        )

    credentials = None

    if GMAIL_TOKEN_FILE.exists():

        credentials = Credentials.from_authorized_user_file(
            GMAIL_TOKEN_FILE,
            GMAIL_SCOPES,
        )

    if not credentials or not credentials.valid:

        if credentials and credentials.expired and credentials.refresh_token:

            credentials.refresh(Request())

        else:

            client_config = {
                "installed": {
                    "client_id": GMAIL_CLIENT_ID,
                    "client_secret": (GMAIL_CLIENT_SECRET),
                    "auth_uri": ("https://accounts.google.com/" "o/oauth2/auth"),
                    "token_uri": ("https://oauth2.googleapis.com/" "token"),
                    "redirect_uris": [
                        "http://localhost",
                    ],
                }
            }

            flow = InstalledAppFlow.from_client_config(
                client_config,
                scopes=GMAIL_SCOPES,
            )

            credentials = flow.run_local_server(
                host="localhost",
                port=0,
                open_browser=True,
            )

        GMAIL_TOKEN_FILE.write_text(
            credentials.to_json(),
            encoding="utf-8",
        )

    return credentials


def get_gmail_service():

    credentials = get_gmail_credentials()

    return build(
        "gmail",
        "v1",
        credentials=credentials,
    )


# =========================================================

# GOOGLE CALENDAR AUTHENTICATION

# =========================================================


def get_calendar_credentials() -> Credentials:
    """

    Load, refresh, or create Google Calendar OAuth credentials.

    """

    if not GMAIL_CLIENT_ID or not GMAIL_CLIENT_SECRET:

        raise RuntimeError(
            "Missing GMAIL_OAUTH_CLIENT_ID or " "GMAIL_OAUTH_CLIENT_SECRET in .env"
        )

    credentials = None

    if CALENDAR_TOKEN_FILE.exists():

        credentials = Credentials.from_authorized_user_file(
            CALENDAR_TOKEN_FILE,
            CALENDAR_SCOPES,
        )

    if not credentials or not credentials.valid:

        if credentials and credentials.expired and credentials.refresh_token:

            credentials.refresh(Request())

        else:

            client_config = {
                "installed": {
                    "client_id": GMAIL_CLIENT_ID,
                    "client_secret": (GMAIL_CLIENT_SECRET),
                    "auth_uri": ("https://accounts.google.com/" "o/oauth2/auth"),
                    "token_uri": ("https://oauth2.googleapis.com/token"),
                    "redirect_uris": [
                        "http://localhost",
                    ],
                }
            }

            flow = InstalledAppFlow.from_client_config(
                client_config,
                scopes=CALENDAR_SCOPES,
            )

            credentials = flow.run_local_server(
                host="localhost",
                port=0,
                open_browser=True,
            )

        CALENDAR_TOKEN_FILE.write_text(
            credentials.to_json(),
            encoding="utf-8",
        )

    return credentials


def get_calendar_service():
    """

    Create authenticated Google Calendar API service.

    """

    credentials = get_calendar_credentials()

    return build(
        "calendar",
        "v3",
        credentials=credentials,
    )


# =========================================================

# GMAIL HELPERS

# =========================================================


def decode_base64url(data: str) -> str:

    if not data:

        return ""

    decoded_bytes = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))

    return decoded_bytes.decode(
        "utf-8",
        errors="replace",
    )


def extract_email_body(payload: dict) -> dict:

    plain_text = ""

    html_text = ""

    mime_type = payload.get(
        "mimeType",
        "",
    )

    body_data = payload.get("body", {}).get("data")

    if body_data:

        decoded = decode_base64url(body_data)

        if mime_type == "text/plain":

            plain_text = decoded

        elif mime_type == "text/html":

            html_text = decoded

    for part in payload.get(
        "parts",
        [],
    ):

        result = extract_email_body(part)

        if not plain_text and result["plain_text"]:

            plain_text = result["plain_text"]

        if not html_text and result["html"]:

            html_text = result["html"]

    return {
        "plain_text": plain_text,
        "html": html_text,
    }


def get_email_headers(payload: dict) -> dict:

    return {
        header["name"].lower(): header["value"]
        for header in payload.get(
            "headers",
            [],
        )
    }


# =========================================================

# GMAIL TOOLS

# =========================================================


@mcp.tool()
def search_emails(
    query: str = "in:inbox",
    max_results: int = 5,
) -> dict:
    """

    Search Gmail using Gmail query syntax.

    """

    if not query.strip():

        return {
            "error": "Search query cannot be empty",
        }

    if max_results < 1 or max_results > 20:

        return {
            "error": ("max_results must be between " "1 and 20"),
        }

    service = get_gmail_service()

    response = (
        service.users()
        .messages()
        .list(
            userId="me",
            q=query,
            maxResults=max_results,
        )
        .execute()
    )

    messages = response.get(
        "messages",
        [],
    )

    emails = []

    for message in messages:

        message_id = message["id"]

        email = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=message_id,
                format="metadata",
                metadataHeaders=[
                    "From",
                    "To",
                    "Subject",
                    "Date",
                ],
            )
            .execute()
        )

        headers = get_email_headers(
            email.get(
                "payload",
                {},
            )
        )

        emails.append(
            {
                "id": email.get("id"),
                "thread_id": email.get("threadId"),
                "from": headers.get(
                    "from",
                    "",
                ),
                "to": headers.get(
                    "to",
                    "",
                ),
                "subject": headers.get(
                    "subject",
                    "",
                ),
                "date": headers.get(
                    "date",
                    "",
                ),
                "snippet": email.get(
                    "snippet",
                    "",
                ),
            }
        )

    return {
        "query": query,
        "count": len(emails),
        "emails": emails,
    }


@mcp.tool()
def get_email(message_id: str) -> dict:
    """

    Get a Gmail message including its body.

    """

    if not message_id.strip():

        return {
            "error": "message_id is required",
        }

    service = get_gmail_service()

    email = (
        service.users()
        .messages()
        .get(
            userId="me",
            id=message_id,
            format="full",
        )
        .execute()
    )

    payload = email.get(
        "payload",
        {},
    )

    headers = get_email_headers(payload)

    body = extract_email_body(payload)

    return {
        "id": email.get("id"),
        "thread_id": email.get("threadId"),
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "cc": headers.get("cc", ""),
        "subject": headers.get(
            "subject",
            "",
        ),
        "date": headers.get("date", ""),
        "snippet": email.get(
            "snippet",
            "",
        ),
        "body": body["plain_text"],
        "html_body": body["html"],
    }


@mcp.tool()
def create_draft(
    to: str,
    subject: str,
    body: str,
    cc: str = "",
    bcc: str = "",
) -> dict:
    """

    Create a Gmail draft.



    This does not send the email.

    """

    if not to.strip():

        return {
            "error": ("Recipient email address is required"),
        }

    if not subject.strip():

        return {
            "error": "Subject is required",
        }

    if not body.strip():

        return {
            "error": "Email body is required",
        }

    message = EmailMessage()

    message["To"] = to.strip()

    message["Subject"] = subject.strip()

    if cc.strip():

        message["Cc"] = cc.strip()

    if bcc.strip():

        message["Bcc"] = bcc.strip()

    message.set_content(body)

    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

    service = get_gmail_service()

    draft = (
        service.users()
        .drafts()
        .create(
            userId="me",
            body={
                "message": {
                    "raw": encoded_message,
                }
            },
        )
        .execute()
    )

    draft_message = draft.get(
        "message",
        {},
    )

    return {
        "success": True,
        "message": ("Gmail draft created successfully"),
        "draft_id": draft.get("id"),
        "message_id": draft_message.get("id"),
        "thread_id": draft_message.get("threadId"),
        "to": to,
        "subject": subject,
    }


# =========================================================

# FETCH HELPERS

# =========================================================


class HTMLTextExtractor(HTMLParser):

    def __init__(self):

        super().__init__()

        self.parts = []

        self.ignore_depth = 0

    def handle_starttag(
        self,
        tag,
        attrs,
    ):

        if tag in {
            "script",
            "style",
            "noscript",
        }:

            self.ignore_depth += 1

    def handle_endtag(self, tag):

        if (
            tag
            in {
                "script",
                "style",
                "noscript",
            }
            and self.ignore_depth > 0
        ):

            self.ignore_depth -= 1

    def handle_data(self, data):

        if self.ignore_depth:

            return

        text = data.strip()

        if text:

            self.parts.append(text)

    def get_text(self):

        return "\n".join(self.parts)


def validate_public_url(
    url: str,
) -> dict | None:
    """

    Block localhost/private/internal addresses.

    """

    parsed = urlparse(url)

    if parsed.scheme not in {
        "http",
        "https",
    }:

        return {
            "error": ("Only http and https URLs " "are allowed"),
        }

    if not parsed.hostname:

        return {
            "error": "Invalid URL",
        }

    hostname = parsed.hostname.lower()

    if hostname == "localhost" or hostname.endswith(".localhost"):

        return {
            "error": ("Localhost URLs are not allowed"),
        }

    try:

        addresses = socket.getaddrinfo(
            hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
        )

    except socket.gaierror:

        return {
            "error": ("Unable to resolve hostname"),
        }

    for address in addresses:

        ip_string = address[4][0]

        try:

            ip = ipaddress.ip_address(ip_string)

        except ValueError:

            continue

        if not ip.is_global:

            return {
                "error": ("Private or internal network " "addresses are not allowed"),
            }

    return None


# =========================================================

# FETCH TOOL

# =========================================================


@mcp.tool()
def fetch_url(
    url: str,
    max_chars: int = MAX_FETCH_CHARS,
) -> dict:
    """

    Fetch readable content from a public HTTP/HTTPS URL.



    Localhost and private/internal network addresses

    are blocked.

    """

    if not url.strip():

        return {
            "error": "URL is required",
        }

    if max_chars < 1000 or max_chars > 50000:

        return {
            "error": ("max_chars must be between " "1000 and 50000"),
        }

    current_url = url.strip()

    try:

        with httpx.Client(
            timeout=15,
            follow_redirects=False,
            headers={"User-Agent": ("multi-mcp-agent/0.1")},
        ) as client:

            for _ in range(6):

                validation_error = validate_public_url(current_url)

                if validation_error:

                    return validation_error

                response = client.get(current_url)

                if response.status_code in {
                    301,
                    302,
                    303,
                    307,
                    308,
                }:

                    location = response.headers.get("location")

                    if not location:

                        return {
                            "error": ("Redirect response " "has no location"),
                        }

                    current_url = urljoin(
                        current_url,
                        location,
                    )

                    continue

                response.raise_for_status()

                content_type = response.headers.get(
                    "content-type",
                    "",
                )

                text = response.text

                if "text/html" in content_type.lower():

                    parser = HTMLTextExtractor()

                    parser.feed(text)

                    text = parser.get_text()

                original_length = len(text)

                truncated = original_length > max_chars

                if truncated:

                    text = text[:max_chars]

                return {
                    "success": True,
                    "url": current_url,
                    "status_code": (response.status_code),
                    "content_type": (content_type),
                    "content": text,
                    "truncated": truncated,
                    "original_length": (original_length),
                }

            return {
                "error": "Too many redirects",
            }

    except httpx.HTTPStatusError as exc:

        return {
            "error": (f"HTTP error: " f"{exc.response.status_code}"),
        }

    except httpx.RequestError as exc:

        return {
            "error": (f"Unable to fetch URL: {exc}"),
        }


# =========================================================

# GIT HELPERS

# =========================================================


def run_git(
    arguments: list[str],
) -> dict:
    """

    Run Git only inside this project's repository.

    """

    try:

        result = subprocess.run(
            [
                "git",
                *arguments,
            ],
            cwd=PROJECT_REPO,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    except subprocess.TimeoutExpired:

        return {
            "success": False,
            "error": "Git command timed out",
        }

    except OSError as exc:

        return {
            "success": False,
            "error": str(exc),
        }

    return {
        "success": (result.returncode == 0),
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def validate_repo_path(
    file_path: str,
) -> bool:
    """

    Prevent Git path arguments from escaping

    the project repository.

    """

    path = (PROJECT_REPO / file_path).resolve()

    try:

        path.relative_to(PROJECT_REPO.resolve())

        return True

    except ValueError:

        return False


# =========================================================

# GIT TOOLS

# =========================================================


@mcp.tool()
def git_status() -> dict:
    """

    Show the Git status of this project.

    """

    result = run_git(
        [
            "status",
            "--short",
            "--branch",
        ]
    )

    result["repository"] = str(PROJECT_REPO)

    return result


@mcp.tool()
def git_log(
    max_count: int = 5,
) -> dict:
    """

    Show recent Git commits.

    """

    if max_count < 1 or max_count > 20:

        return {
            "error": ("max_count must be between " "1 and 20"),
        }

    return run_git(
        [
            "log",
            f"-n{max_count}",
            ("--pretty=format:" "%h | %ad | %an | %s"),
            "--date=iso",
        ]
    )


@mcp.tool()
def git_diff(
    staged: bool = False,
) -> dict:
    """

    Show unstaged or staged Git differences.

    """

    arguments = [
        "diff",
    ]

    if staged:

        arguments.append("--cached")

    return run_git(arguments)


@mcp.tool()
def git_show(
    ref: str = "HEAD",
) -> dict:
    """

    Show a Git commit or reference.

    """

    if not ref.strip():

        return {
            "error": "Git ref is required",
        }

    return run_git(
        [
            "show",
            "--stat",
            "--oneline",
            ref.strip(),
        ]
    )


@mcp.tool()
def git_branch() -> dict:
    """

    List Git branches.

    """

    return run_git(
        [
            "branch",
            "-vv",
        ]
    )


@mcp.tool()
def git_add(
    paths: list[str],
) -> dict:
    """

    Stage specific files.



    This modifies the Git index.

    """

    if not paths:

        return {
            "error": ("At least one path is required"),
        }

    safe_paths = []

    for path in paths:

        if not path.strip():

            continue

        if not validate_repo_path(path):

            return {
                "error": (f"Path is outside repository: " f"{path}"),
            }

        safe_paths.append(path)

    if not safe_paths:

        return {
            "error": ("No valid paths were provided"),
        }

    return run_git(
        [
            "add",
            "--",
            *safe_paths,
        ]
    )


@mcp.tool()
def git_commit(
    message: str,
) -> dict:
    """

    Commit already staged changes.



    This does not automatically stage files.

    """

    if not message.strip():

        return {
            "error": ("Commit message is required"),
        }

    return run_git(
        [
            "commit",
            "-m",
            message.strip(),
        ]
    )


@mcp.tool()
def git_create_branch(
    branch_name: str,
) -> dict:
    """

    Create a new Git branch without switching to it.

    """

    if not branch_name.strip():

        return {
            "error": ("Branch name is required"),
        }

    validation = run_git(
        [
            "check-ref-format",
            "--branch",
            branch_name.strip(),
        ]
    )

    if not validation["success"]:

        return {
            "error": ("Invalid Git branch name"),
        }

    return run_git(
        [
            "branch",
            branch_name.strip(),
        ]
    )


@mcp.tool()
def git_checkout(
    branch_name: str,
) -> dict:
    """

    Switch to an existing Git branch.

    """

    if not branch_name.strip():

        return {
            "error": ("Branch name is required"),
        }

    return run_git(
        [
            "checkout",
            branch_name.strip(),
        ]
    )


# =========================================================

# CALENDAR DATETIME VALIDATION

# =========================================================


def validate_calendar_event_times(
    start_datetime: str,
    end_datetime: str,
    timezone: str,
) -> dict | None:
    """
    Validate Calendar event start/end datetimes.

    Returns None when valid.
    Returns an error dictionary when invalid.
    """

    try:
        calendar_timezone = ZoneInfo(timezone)

    except Exception:
        return {
            "success": False,
            "error": (f"Invalid timezone: {timezone}"),
        }

    try:
        parsed_start = datetime.fromisoformat(
            start_datetime.replace(
                "Z",
                "+00:00",
            )
        )

    except ValueError:
        return {
            "success": False,
            "error": ("start_datetime must be a valid " "ISO8601 datetime"),
        }

    try:
        parsed_end = datetime.fromisoformat(
            end_datetime.replace(
                "Z",
                "+00:00",
            )
        )

    except ValueError:
        return {
            "success": False,
            "error": ("end_datetime must be a valid " "ISO8601 datetime"),
        }

    if parsed_start.tzinfo is None:
        return {
            "success": False,
            "error": ("start_datetime must include " "a timezone offset"),
        }

    if parsed_end.tzinfo is None:
        return {
            "success": False,
            "error": ("end_datetime must include " "a timezone offset"),
        }

    start_in_timezone = parsed_start.astimezone(calendar_timezone)

    end_in_timezone = parsed_end.astimezone(calendar_timezone)

    current_datetime = datetime.now(calendar_timezone)

    if end_in_timezone <= start_in_timezone:
        return {
            "success": False,
            "error": ("Calendar event end time must " "be after the start time"),
        }

    if start_in_timezone < current_datetime:
        return {
            "success": False,
            "error": (
                "Cannot create a calendar event "
                "in the past. "
                f"Current datetime is "
                f"{current_datetime.isoformat()}."
            ),
        }

    return None


# =========================================================

# GOOGLE CALENDAR TOOLS

# =========================================================


@mcp.tool()
def list_calendar_events(
    time_min: str,
    time_max: str,
    max_results: int = 10,
) -> dict:
    """

    List Google Calendar events between two RFC3339 datetimes.



    Example:

    time_min = 2026-10-07T00:00:00+05:45

    time_max = 2026-10-08T00:00:00+05:45

    """

    if not time_min.strip():

        return {
            "error": "time_min is required",
        }

    if not time_max.strip():

        return {
            "error": "time_max is required",
        }

    if max_results < 1 or max_results > 50:

        return {
            "error": ("max_results must be between 1 and 50"),
        }

    try:

        service = get_calendar_service()

        response = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )

        events = []

        for event in response.get(
            "items",
            [],
        ):

            start = event.get(
                "start",
                {},
            )

            end = event.get(
                "end",
                {},
            )

            events.append(
                {
                    "id": event.get("id"),
                    "summary": event.get(
                        "summary",
                        "(No title)",
                    ),
                    "description": event.get(
                        "description",
                        "",
                    ),
                    "location": event.get(
                        "location",
                        "",
                    ),
                    "start": (start.get("dateTime") or start.get("date")),
                    "end": (end.get("dateTime") or end.get("date")),
                    "status": event.get("status"),
                    "html_link": event.get("htmlLink"),
                }
            )

        return {
            "success": True,
            "count": len(events),
            "events": events,
        }

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc),
        }


@mcp.tool()
def create_calendar_event(
    summary: str,
    start_datetime: str,
    end_datetime: str,
    description: str = "",
    location: str = "",
    timezone: str = CALENDAR_TIMEZONE,
) -> dict:
    """
    Create an event in the user's primary Google Calendar.

    Datetimes must be RFC3339 / ISO8601 and include
    a timezone offset.

    Example:
    2026-10-07T15:00:00+05:45
    """

    if not summary.strip():
        return {
            "success": False,
            "error": ("Event summary is required"),
        }

    if not start_datetime.strip():
        return {
            "success": False,
            "error": ("start_datetime is required"),
        }

    if not end_datetime.strip():
        return {
            "success": False,
            "error": ("end_datetime is required"),
        }

    validation_error = validate_calendar_event_times(
        start_datetime,
        end_datetime,
        timezone,
    )

    if validation_error:
        return validation_error

    event_body = {
        "summary": summary.strip(),
        "description": (description.strip()),
        "location": location.strip(),
        "start": {
            "dateTime": start_datetime,
            "timeZone": timezone,
        },
        "end": {
            "dateTime": end_datetime,
            "timeZone": timezone,
        },
    }

    try:
        service = get_calendar_service()

        event = (
            service.events()
            .insert(
                calendarId="primary",
                body=event_body,
            )
            .execute()
        )

        return {
            "success": True,
            "message": ("Calendar event created " "successfully"),
            "event_id": event.get("id"),
            "summary": event.get("summary"),
            "start": (
                event.get(
                    "start",
                    {},
                ).get("dateTime")
            ),
            "end": (
                event.get(
                    "end",
                    {},
                ).get("dateTime")
            ),
            "timezone": timezone,
            "html_link": event.get("htmlLink"),
        }

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
        }


@mcp.tool()
def update_calendar_event(
    event_id: str,
    summary: str = "",
    start_datetime: str = "",
    end_datetime: str = "",
    description: str = "",
    location: str = "",
    timezone: str = CALENDAR_TIMEZONE,
) -> dict:
    """

    Update an existing Google Calendar event.

    """

    if not event_id.strip():

        return {
            "success": False,
            "error": "event_id is required",
        }

    try:

        service = get_calendar_service()

        event = (
            service.events()
            .get(
                calendarId="primary",
                eventId=event_id,
            )
            .execute()
        )

        if summary.strip():

            event["summary"] = summary.strip()

        if description.strip():

            event["description"] = description.strip()

        if location.strip():

            event["location"] = location.strip()

        if start_datetime.strip():

            event["start"] = {
                "dateTime": start_datetime,
                "timeZone": timezone,
            }

        if end_datetime.strip():

            event["end"] = {
                "dateTime": end_datetime,
                "timeZone": timezone,
            }

        updated = (
            service.events()
            .update(
                calendarId="primary",
                eventId=event_id,
                body=event,
            )
            .execute()
        )

        return {
            "success": True,
            "message": ("Calendar event updated successfully"),
            "event_id": updated.get("id"),
            "summary": updated.get("summary"),
            "start": updated.get(
                "start",
                {},
            ).get(
                "dateTime",
                updated.get(
                    "start",
                    {},
                ).get("date"),
            ),
            "end": updated.get(
                "end",
                {},
            ).get(
                "dateTime",
                updated.get(
                    "end",
                    {},
                ).get("date"),
            ),
            "html_link": updated.get("htmlLink"),
        }

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc),
        }


@mcp.tool()
def delete_calendar_event(
    event_id: str,
) -> dict:
    """

    Delete an event from the user's primary Google Calendar.

    """

    if not event_id.strip():

        return {
            "success": False,
            "error": "event_id is required",
        }

    try:

        service = get_calendar_service()

        (
            service.events()
            .delete(
                calendarId="primary",
                eventId=event_id,
            )
            .execute()
        )

        return {
            "success": True,
            "message": ("Calendar event deleted successfully"),
            "event_id": event_id,
        }

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc),
        }

# =========================================================
# RAG TOOLS
# =========================================================


@mcp.tool()
def index_document(
    file_path: str,
    thread_id: str,
) -> dict:
    """
    Index an uploaded PDF for the current conversation.

    The PDF must already exist inside the project's
    .rag/uploads directory.

    The document is extracted page-by-page, chunked,
    embedded, and persisted for later retrieval.
    """

    return rag_index_pdf(
        file_path=file_path,
        thread_id=thread_id,
    )


@mcp.tool()
def search_documents(
    query: str,
    thread_id: str,
    document_ids: list[str] | None = None,
    top_k: int = 6,
) -> dict:
    """
    Search indexed PDFs for evidence relevant to a question.

    Uses hybrid retrieval:
    semantic embeddings + BM25 + reciprocal rank fusion
    + MMR reranking.

    Results include filename and page-number citations.
    """

    return rag_search_documents(
        query=query,
        thread_id=thread_id,
        document_ids=document_ids,
        top_k=top_k,
    )


@mcp.tool()
def list_documents(
    thread_id: str,
) -> dict:
    """
    List PDFs indexed for the current conversation.
    """

    return rag_list_documents(
        thread_id=thread_id,
    )


@mcp.tool()
def delete_document(
    document_id: str,
    thread_id: str,
) -> dict:
    """
    Remove a PDF from the RAG index for this conversation.
    """

    return rag_delete_document(
        document_id=document_id,
        thread_id=thread_id,
    )


# =========================================================

# YOUTUBE TOOL

# =========================================================


@mcp.tool()
def play_youtube_song(song_name: str) -> dict:
    """

    Search YouTube for a song and open the first matching

    result in the default web browser.



    This tool does not download the video or audio.

    """

    if not song_name.strip():

        return {
            "success": False,
            "error": "Song name is required",
        }

    search_query = f"ytsearch1:{song_name.strip()}"

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
    }

    try:

        with YoutubeDL(options) as ydl:

            result = ydl.extract_info(
                search_query,
                download=False,
            )

        entries = result.get(
            "entries",
            [],
        )

        if not entries:

            return {
                "success": False,
                "error": "No matching YouTube video found",
            }

        video = entries[0]

        video_id = video.get("id")

        title = video.get(
            "title",
            song_name,
        )

        if not video_id:

            return {
                "success": False,
                "error": ("Unable to determine YouTube video ID"),
            }

        video_url = "https://www.youtube.com/watch" f"?v={video_id}&autoplay=1"

        opened = webbrowser.open(
            video_url,
            new=2,
        )

        return {
            "success": opened,
            "message": (
                "YouTube video opened in browser"
                if opened
                else "Unable to open browser"
            ),
            "song": song_name,
            "title": title,
            "video_id": video_id,
            "url": video_url,
        }

    except Exception as exc:

        return {
            "success": False,
            "error": (f"Unable to search YouTube: {exc}"),
        }


# =========================================================

# MCP RESOURCE

# =========================================================


@mcp.resource("multi-mcp://about")
def about_server() -> str:

    return (
        "Multi MCP Server providing Weather, Gmail, Google Calendar, "
        "Fetch, Git, and YouTube tools."
    )


# =========================================================

# Run MCP Server

# =========================================================


if __name__ == "__main__":

    mcp.run()
