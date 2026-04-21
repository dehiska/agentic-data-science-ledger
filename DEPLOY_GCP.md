# Deploying Agentic DS Ledger to GCP

This guide walks through a complete Phase 2 deployment: two Cloud Run services (backend + frontend) built via Cloud Build, images stored in Artifact Registry, secrets in Secret Manager, and Supabase as the database.

---

## Prerequisites

- GCP project with billing enabled
- `gcloud` CLI installed and authenticated (`gcloud auth login`)
- Docker installed locally (only needed for local testing, not for Cloud Build)
- A Supabase project created at https://supabase.com
- An OpenAI API key (optional, for Agent Plan feature)

---

## Step 1 — Enable GCP APIs

```bash
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com
```

---

## Step 2 — Create Artifact Registry Repository

```bash
gcloud artifacts repositories create agentic-ledger \
  --repository-format=docker \
  --location=us-central1 \
  --description="Agentic DS Ledger images"
```

Verify:
```bash
gcloud artifacts repositories list --location=us-central1
```

---

## Step 3 — Set Up Supabase Schema

1. Go to https://app.supabase.com → your project → **SQL Editor**
2. Paste and run the full contents of `supabase_setup.sql` from this repo
3. Note your **Project URL** and **anon key** from *Settings → API*

---

## Step 4 — Store Secrets in Secret Manager

```bash
PROJECT_ID=$(gcloud config get-value project)

# Supabase
echo -n "https://your-project.supabase.co" | \
  gcloud secrets create SUPABASE_URL --data-file=- --replication-policy=automatic

echo -n "your-supabase-anon-key" | \
  gcloud secrets create SUPABASE_KEY --data-file=- --replication-policy=automatic

# OpenAI (optional)
echo -n "sk-..." | \
  gcloud secrets create OPENAI_API_KEY --data-file=- --replication-policy=automatic
```

Grant Cloud Build access to the secrets:
```bash
CB_SA=$(gcloud projects describe $PROJECT_ID \
  --format="value(projectNumber)")@cloudbuild.gserviceaccount.com

for SECRET in SUPABASE_URL SUPABASE_KEY OPENAI_API_KEY; do
  gcloud secrets add-iam-policy-binding $SECRET \
    --member="serviceAccount:$CB_SA" \
    --role="roles/secretmanager.secretAccessor"
done
```

Also grant Cloud Build the Cloud Run Admin and Service Account User roles (needed to deploy):
```bash
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$CB_SA" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$CB_SA" \
  --role="roles/iam.serviceAccountUser"
```

---

## Step 5 — Push Code to GitHub

The Cloud Build trigger watches a GitHub repo. Make sure the `phase2-gcp-multiuser` branch is pushed:

```bash
git push origin phase2-gcp-multiuser
```

---

## Step 6 — Create Cloud Build Trigger

### Option A — Console

1. Go to **Cloud Build → Triggers → Create Trigger**
2. Connect your GitHub repo if not already connected
3. Settings:
   - **Name:** `deploy-agentic-ledger`
   - **Event:** Push to branch
   - **Branch:** `^phase2-gcp-multiuser$`
   - **Configuration:** Cloud Build config file → `cloudbuild.yaml`
4. Add substitution variables:

| Variable | Value |
|----------|-------|
| `_REGION` | `us-central1` |
| `_REPO` | `agentic-ledger` |
| `_APP_MODE` | `cloud` |
| `_SUPABASE_URL` | *(your Supabase URL)* |
| `_SUPABASE_KEY` | *(your Supabase anon key)* |
| `_OPENAI_API_KEY` | *(your OpenAI key, or leave blank)* |

### Option B — gcloud CLI

```bash
gcloud builds triggers create github \
  --repo-name=YOUR_GITHUB_REPO \
  --repo-owner=YOUR_GITHUB_USER \
  --branch-pattern="^phase2-gcp-multiuser$" \
  --build-config=cloudbuild.yaml \
  --substitutions=_REGION=us-central1,_REPO=agentic-ledger,_APP_MODE=cloud,\
_SUPABASE_URL=https://your-project.supabase.co,\
_SUPABASE_KEY=your-anon-key,\
_OPENAI_API_KEY=sk-...
```

---

## Step 7 — Trigger a Build

Push a commit to `phase2-gcp-multiuser` (or click **Run** on the trigger):

```bash
git commit --allow-empty -m "trigger: deploy phase2"
git push origin phase2-gcp-multiuser
```

Watch progress at **Cloud Build → History**. The pipeline:
1. Builds backend + frontend images in parallel (~3–5 min)
2. Pushes both images to Artifact Registry
3. Deploys backend Cloud Run service
4. Fetches backend URL → deploys frontend with `BACKEND_URL` set

Total build time: ~8–12 minutes on first run.

---

## Step 8 — Get the Frontend URL

```bash
gcloud run services describe agentic-ledger-frontend \
  --region=us-central1 \
  --format="value(status.url)"
```

Open that URL in your browser — you should see the Agentic DS Ledger UI in cloud mode.

---

## Step 9 — Verify Deployment

```bash
BACKEND=$(gcloud run services describe agentic-ledger-backend \
  --region=us-central1 --format="value(status.url)")

# Health check
curl $BACKEND/health
# Expected: {"status":"ok","mode":"cloud"}

# List projects (should be empty initially)
curl $BACKEND/projects
```

---

## Updating the App

Every push to `phase2-gcp-multiuser` triggers a new build and rolling deploy with zero downtime.

To promote to `main` after testing:
```bash
git checkout main
git merge phase2-gcp-multiuser
git push origin main
# Update trigger branch pattern to ^main$ or create a separate trigger
```

---

## Cost Estimates (us-central1)

| Resource | Spec | Estimated Monthly Cost |
|----------|------|----------------------|
| Cloud Run backend | 1 vCPU, 1 GiB, min 0, max 3 | ~$0–5 (idle → moderate traffic) |
| Cloud Run frontend | 1 vCPU, 1 GiB, min 0, max 3 | ~$0–5 |
| Artifact Registry | ~500 MB images | ~$0.05 |
| Cloud Build | First 120 min/day free | ~$0 |
| Supabase | Free tier (500 MB DB) | $0 |
| **Total** | | **~$0–10/month** |

---

## Troubleshooting

**Build fails: permission denied pushing to Artifact Registry**
```bash
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$CB_SA" \
  --role="roles/artifactregistry.writer"
```

**Frontend shows "Cannot connect to backend"**
- Check `BACKEND_URL` env var on the frontend service:
  ```bash
  gcloud run services describe agentic-ledger-frontend \
    --region=us-central1 --format="yaml" | grep BACKEND_URL
  ```
- Must be the full Cloud Run URL (no trailing slash).

**Supabase connection error**
- Verify `SUPABASE_URL` and `SUPABASE_KEY` are set correctly on the backend service.
- Run `supabase_setup.sql` in the Supabase SQL editor if tables don't exist.

**Container crashes on startup (port error)**
- Both Dockerfiles use `${PORT:-8080}` — Cloud Run injects `$PORT` at runtime.
- Do not hardcode port 8080 in CMD; always use the shell form with the variable.
