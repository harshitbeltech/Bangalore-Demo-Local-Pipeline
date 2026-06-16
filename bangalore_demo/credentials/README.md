# credentials/

Drop your Google service-account key here as `service_account.json` (gitignored).

1. GCP console -> create a service account -> enable **Google Drive API** + **Google Sheets API**.
2. Create a JSON key for it, download, save as `service_account.json` in this folder.
3. Share your evidence Drive folder AND your Google Sheet with the service account's email
   (looks like `something@your-project.iam.gserviceaccount.com`) as **Editor**.
4. Put the Drive folder ID and Sheet ID into `../config/settings.yaml`.

Without this file the pipeline still runs, writing evidence to `../output/evidence/` and rows to
`../output/violations.csv` (local fallback mode).
