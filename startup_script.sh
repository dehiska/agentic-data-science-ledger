#!/bin/bash
# GCP Compute Engine startup script for Phase 2 deployment.
# Deploy with:
#   gcloud compute instances create agentic-ds-ledger \
#     --machine-type=n1-standard-2 --image-family=debian-11 \
#     --image-project=debian-cloud --tags=http-server,https-server \
#     --metadata-from-file startup-script=startup_script.sh

set -e

apt-get update
apt-get install -y python3-pip python3-venv git nginx

# Clone the repo
git clone https://github.com/YOUR_USERNAME/agentic-ds-ledger.git /app
cd /app

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables (replace with real values)
export SUPABASE_URL="${SUPABASE_URL}"
export SUPABASE_KEY="${SUPABASE_KEY}"
export GITHUB_TOKEN="${GITHUB_TOKEN}"
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export APP_MODE="cloud"
export BACKEND_URL="http://localhost:8000"

# Start FastAPI backend (background)
nohup uvicorn backend.api:app --host 127.0.0.1 --port 8000 &

# Start Streamlit frontend on port 8501
nohup streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0 &

# Nginx reverse proxy config
cat > /etc/nginx/sites-available/default <<'EOF'
server {
    listen 80;
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
    location /api/ {
        rewrite ^/api/(.*) /$1 break;
        proxy_pass http://127.0.0.1:8000;
    }
}
EOF

systemctl restart nginx
