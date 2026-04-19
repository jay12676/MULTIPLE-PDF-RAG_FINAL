@echo off
cd /d C:\Users\LOQ\Desktop\Multiple_Pdf_Upload_RAG

echo Starting DocuMind servers...

start /b python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info > logs\backend.log 2>&1
start /b python -m celery -A app.workers.celery_app worker --loglevel=info --pool=threads --concurrency=2 > logs\celery.log 2>&1
start /b python -m streamlit run frontend/app.py --server.port 8501 > logs\streamlit.log 2>&1

echo All servers started. Open http://localhost:8501
