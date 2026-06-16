"""Google Drive + Sheets client with a transparent local fallback.

If a service-account key is present and the google libs import, evidence images are
uploaded to a Drive folder (returning anyone-with-link view URLs) and rows are appended
to a Google Sheet. Otherwise it silently falls back to writing images under
output/evidence/ and returns a local file path — the rest of the pipeline is unchanged.
"""
import io
import logging
import os
import threading

logger = logging.getLogger("bangalore.drive")

_SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]


class DriveSheets:
    def __init__(self, settings: dict):
        d = settings["drive"]
        self.folder_id = d.get("evidence_folder_id", "") or None
        self.sheet_id = d.get("sheet_id", "") or None
        self.sheet_tab = d.get("sheet_tab", "violations")
        self.share_email = d.get("share_email", "") or None
        self.local_only = True
        # googleapiclient/httplib2 is NOT thread-safe; serialize all Drive/Sheets HTTP
        # so concurrent violation workers can't corrupt the shared TLS connection
        # ("SSL: WRONG_VERSION_NUMBER" / read timeouts).
        self._http_lock = threading.Lock()
        self.evidence_dir = os.path.join(settings["output"]["dir"], "evidence")
        os.makedirs(self.evidence_dir, exist_ok=True)

        # Prefer OAuth user credentials. A service account cannot own files in a
        # personal Gmail Drive ("Service Accounts do not have storage quota"), so
        # uploads/sheet-create fail there; OAuth files are owned by the user.
        creds = self._load_credentials(d)
        if creds is not None:
            try:
                self._build(creds)
                self.local_only = False
                logger.info("Google Drive/Sheets enabled.")
            except Exception as e:
                logger.warning(f"Drive/Sheets init failed ({e}); using local fallback.")
        else:
            logger.info("No usable Google credentials; running in LOCAL FALLBACK mode "
                        f"(evidence -> {self.evidence_dir}).")

    # ── credential loading ─────────────────────────────────────────────────────
    def _load_credentials(self, d: dict):
        token = d.get("oauth_token", "")
        if token and os.path.exists(token):
            try:
                return self._load_oauth_token(token)
            except Exception as e:
                logger.warning(f"OAuth token unusable ({e}); run `python authorize.py`.")
        sa = d.get("service_account", "")
        if sa and os.path.exists(sa):
            try:
                from google.oauth2.service_account import Credentials
                return Credentials.from_service_account_file(sa, scopes=_SCOPES)
            except Exception as e:
                logger.warning(f"Service-account load failed ({e}).")
        return None

    def _load_oauth_token(self, token_path: str):
        from google.oauth2.credentials import Credentials as UserCredentials
        from google.auth.transport.requests import Request
        creds = UserCredentials.from_authorized_user_file(token_path, _SCOPES)
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                with open(token_path, "w") as f:
                    f.write(creds.to_json())
            else:
                raise RuntimeError("token invalid and not refreshable")
        return creds

    # ── Google init ──────────────────────────────────────────────────────────
    def _build(self, creds):
        from googleapiclient.discovery import build
        self.drive = build("drive", "v3", credentials=creds, cache_discovery=False)
        self.sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
        if not self.sheet_id:
            self.sheet_id = self._create_sheet(creds)

    def _create_sheet(self, creds) -> str:
        body = {"properties": {"title": "Bangalore Demo — Violations"}}
        sh = self.sheets.spreadsheets().create(body=body, fields="spreadsheetId").execute()
        sid = sh["spreadsheetId"]
        # make viewable by anyone with the link (demo convenience)
        try:
            self.drive.permissions().create(
                fileId=sid, body={"type": "anyone", "role": "reader"}).execute()
        except Exception:
            pass
        if self.share_email:
            try:
                self.drive.permissions().create(
                    fileId=sid, sendNotificationEmail=False,
                    body={"type": "user", "role": "writer",
                          "emailAddress": self.share_email}).execute()
            except Exception:
                pass
        logger.info(f"Created sheet: https://docs.google.com/spreadsheets/d/{sid}")
        return sid

    # ── Uploads ──────────────────────────────────────────────────────────────
    def upload_jpeg(self, jpeg_bytes: bytes, filename: str) -> str:
        """Return a shareable URL (Drive) or a local path (fallback)."""
        if self.local_only:
            path = os.path.join(self.evidence_dir, filename)
            with open(path, "wb") as f:
                f.write(jpeg_bytes)
            return path
        from googleapiclient.http import MediaIoBaseUpload
        meta = {"name": filename}
        if self.folder_id:
            meta["parents"] = [self.folder_id]
        media = MediaIoBaseUpload(io.BytesIO(jpeg_bytes), mimetype="image/jpeg")
        with self._http_lock:
            f = self.drive.files().create(body=meta, media_body=media, fields="id").execute()
            fid = f["id"]
            try:
                self.drive.permissions().create(
                    fileId=fid, body={"type": "anyone", "role": "reader"}).execute()
            except Exception:
                pass
        return f"https://drive.google.com/file/d/{fid}/view?usp=drive_link"

    # ── Sheet rows ───────────────────────────────────────────────────────────
    def append_row(self, values: list):
        if self.local_only:
            return
        try:
            with self._http_lock:
                self.sheets.spreadsheets().values().append(
                    spreadsheetId=self.sheet_id,
                    range=f"{self.sheet_tab}!A1",
                    valueInputOption="USER_ENTERED",
                    insertDataOption="INSERT_ROWS",
                    body={"values": [values]},
                ).execute()
        except Exception as e:
            logger.warning(f"Sheet append failed: {e}")

    def ensure_header(self, header: list):
        if self.local_only:
            return
        try:
            got = self.sheets.spreadsheets().values().get(
                spreadsheetId=self.sheet_id, range=f"{self.sheet_tab}!A1:Z1").execute()
            if got.get("values"):
                return
        except Exception:
            # tab may not exist yet — create it
            try:
                self.sheets.spreadsheets().batchUpdate(
                    spreadsheetId=self.sheet_id,
                    body={"requests": [{"addSheet": {"properties": {"title": self.sheet_tab}}}]},
                ).execute()
            except Exception:
                pass
        self.append_row(header)
