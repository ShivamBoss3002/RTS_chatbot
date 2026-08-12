@echo off
cd /d D:\AppData\chatbot_rts
call .venv\Scripts\activate
uvicorn app:app --host 0.0.0.0 --port 7000