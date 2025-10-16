@echo off
echo Starting Backend and Frontend...

start "Backend" cmd /c "cd backend && pip install -r requirements.txt && python start_server.py"
start "Frontend" cmd /c "cd frontend && npm i && npm start"

echo Both servers started!
exit