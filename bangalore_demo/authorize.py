#!/usr/bin/env python3
"""One-time Google OAuth consent -> writes credentials/token.json.

Files created by the pipeline (evidence images, the violations sheet) will be owned
by the Google account you authorize here and count against its 15GB. This is the
correct path for a personal Gmail account (a service account cannot own files there).

Prereq: a Desktop OAuth client downloaded as credentials/client_secret.json
  (GCP console -> APIs & Services -> Credentials -> Create credentials ->
   OAuth client ID -> Application type: Desktop app -> download JSON).

Run (no SSH port-forwarding needed):
    source /home/cv-gpu-2/harshit_workspace/.venv/bin/activate
    python authorize.py

It prints a URL. Open it in your laptop browser, sign in, approve. The browser will then
try to load http://localhost:8765/?code=... and show "site can't be reached" — that is
EXPECTED. Copy the full URL from the address bar and paste it back into the terminal.
"""
import core._env  # noqa: F401

import os
import sys

from core.config import load_settings

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]


def main():
    settings = load_settings("config/settings.yaml")
    d = settings["drive"]
    client = d.get("oauth_client", "")
    token = d.get("oauth_token", "")

    if not client or not os.path.exists(client):
        sys.exit(f"OAuth client file not found: {client!r}\n"
                 "Download a Desktop OAuth client JSON to credentials/client_secret.json first.")

    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(client, SCOPES)
    # Runs a one-shot listener on the server's port 8765. With an SSH tunnel from your
    # laptop (ssh -L 8765:localhost:8765 ...), the browser's redirect to localhost:8765
    # reaches this listener and the code is captured automatically — no copy-paste.
    creds = flow.run_local_server(
        host="localhost", port=8765, open_browser=False,
        authorization_prompt_message=(
            "\nOpen this URL in your laptop browser and approve "
            "(sign in as harshit@beltech.ai):\n\n{url}\n\n"
            "Waiting for you to approve…\n"),
        success_message=("Authorized! You can close this browser tab "
                         "and return to the terminal."))

    os.makedirs(os.path.dirname(token), exist_ok=True)
    with open(token, "w") as f:
        f.write(creds.to_json())
    print(f"\n✅ Wrote {token}")
    print("Drive/Sheets are now live. Verify with:  python verify_drive.py")


if __name__ == "__main__":
    main()
