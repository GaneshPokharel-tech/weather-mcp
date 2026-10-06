import base64
import os
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from mcp.server import MCPServer

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()

CLIENT_ID = os.getenv("GMAIL_OAUTH_CLIENT_ID")
CLIENT_SECRET = os.getenv("GMAIL_OAUTH_CLIENT_SECRET")

TOKEN_FILE = Path(".gmail_token.json")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]


# ---------------------------------------------------------
# MCP Server
# ---------------------------------------------------------

mcp = MCPServer("Gmail MCP")


# ---------------------------------------------------------
# Gmail Authentication
# ---------------------------------------------------------


def get_gmail_credentials() -> Credentials:
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError(
            "Missing GMAIL_OAUTH_CLIENT_ID or " "GMAIL_OAUTH_CLIENT_SECRET in .env"
        )

    credentials = None

    if TOKEN_FILE.exists():
        credentials = Credentials.from_authorized_user_file(
            TOKEN_FILE,
            SCOPES,
        )

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())

        else:
            client_config = {
                "installed": {
                    "client_id": CLIENT_ID,
                    "client_secret": CLIENT_SECRET,
                    "auth_uri": ("https://accounts.google.com/o/oauth2/auth"),
                    "token_uri": ("https://oauth2.googleapis.com/token"),
                    "redirect_uris": [
                        "http://localhost",
                    ],
                }
            }

            flow = InstalledAppFlow.from_client_config(
                client_config,
                scopes=SCOPES,
            )

            credentials = flow.run_local_server(
                host="localhost",
                port=0,
                open_browser=True,
            )

        TOKEN_FILE.write_text(
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


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def decode_base64url(data: str) -> str:
    if not data:
        return ""

    decoded_bytes = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))

    return decoded_bytes.decode(
        "utf-8",
        errors="replace",
    )


def extract_body(payload: dict) -> dict:
    """
    Extract plain-text and HTML bodies from a Gmail MIME payload.
    """

    plain_text = ""
    html_text = ""

    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")

    if body_data:
        decoded = decode_base64url(body_data)

        if mime_type == "text/plain":
            plain_text = decoded

        elif mime_type == "text/html":
            html_text = decoded

    for part in payload.get("parts", []):
        result = extract_body(part)

        if not plain_text and result["plain_text"]:
            plain_text = result["plain_text"]

        if not html_text and result["html"]:
            html_text = result["html"]

    return {
        "plain_text": plain_text,
        "html": html_text,
    }


def get_headers(payload: dict) -> dict:
    return {
        header["name"].lower(): header["value"] for header in payload.get("headers", [])
    }


# ---------------------------------------------------------
# Gmail MCP Tools
# ---------------------------------------------------------


@mcp.tool()
def search_emails(
    query: str = "in:inbox",
    max_results: int = 5,
) -> dict:
    """
    Search Gmail messages using Gmail search syntax.

    Examples:
    - in:inbox is:unread
    - from:user@example.com
    - subject:invoice
    - newer_than:7d
    """

    if not query.strip():
        return {
            "error": "Search query cannot be empty",
        }

    if max_results < 1 or max_results > 20:
        return {
            "error": "max_results must be between 1 and 20",
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

    messages = response.get("messages", [])

    if not messages:
        return {
            "query": query,
            "count": 0,
            "emails": [],
        }

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

        headers = get_headers(email.get("payload", {}))

        emails.append(
            {
                "id": email.get("id"),
                "thread_id": email.get("threadId"),
                "from": headers.get("from", ""),
                "to": headers.get("to", ""),
                "subject": headers.get("subject", ""),
                "date": headers.get("date", ""),
                "snippet": email.get("snippet", ""),
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
    Retrieve a Gmail message by its message ID,
    including sender, recipients, subject, date,
    snippet, and message body.
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

    payload = email.get("payload", {})

    headers = get_headers(payload)

    body = extract_body(payload)

    return {
        "id": email.get("id"),
        "thread_id": email.get("threadId"),
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "cc": headers.get("cc", ""),
        "subject": headers.get("subject", ""),
        "date": headers.get("date", ""),
        "snippet": email.get("snippet", ""),
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

    This tool only creates a draft.
    It does not send the email.
    """

    if not to.strip():
        return {
            "error": "Recipient email address is required",
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

    draft_message = draft.get("message", {})

    return {
        "success": True,
        "message": "Gmail draft created successfully",
        "draft_id": draft.get("id"),
        "message_id": draft_message.get("id"),
        "thread_id": draft_message.get("threadId"),
        "to": to,
        "subject": subject,
    }


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
