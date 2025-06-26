@echo off
REM Build script for Agentic RAG application on Windows

echo 🐋 Building Agentic RAG Docker Images
echo ======================================

REM Check if .env exists
if not exist .env (
    echo ⚠️  .env file not found. Copying from .env.example...
    copy .env.example .env
    echo 📝 Please edit .env file with your Azure configuration before running.
    exit /b 1
)

REM Build backend
echo 🏗️  Building backend image...
docker build -t agenticrag-backend:latest ./backend

REM Build frontend
echo 🏗️  Building frontend image...
docker build -t agenticrag-frontend:latest ./frontend

echo ✅ Build completed successfully!
echo.
echo 🚀 To start the application:
echo    docker-compose up -d
echo.
echo 📊 To view logs:
echo    docker-compose logs -f
echo.
echo 🌐 Access URLs:
echo    Frontend: http://localhost
echo    Backend:  http://localhost:8000
echo    API Docs: http://localhost:8000/docs
