# Phishing Detector Launch Script

echo "🚀 Starting Phishing Detector API..."
Start-Process powershell -ArgumentList "-NoExit -Command venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000"

echo "🌐 Starting Next.js Web App..."
Set-Location web
npm run dev
