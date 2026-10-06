@echo off
REM Starts the retrieval demo. Open http://localhost:8000 (other devices on your Wi-Fi: http://<this-PC-IP>:8000)
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" deep_learning
uvicorn server:app --host 0.0.0.0 --port 8000
pause
