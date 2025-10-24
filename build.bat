@echo off
REM Build script for BEN2 Serverless CPU Worker (Windows)
REM Usage: build.bat [tag]

setlocal

set IMAGE_NAME=ben2-serverless-cpu
set TAG=%1
if "%TAG%"=="" set TAG=latest
set FULL_IMAGE=%IMAGE_NAME%:%TAG%

echo ======================================
echo Building BEN2 Serverless CPU Worker
echo ======================================
echo Image: %FULL_IMAGE%
echo Build context: Parent directory (..)
echo.

REM Check if parent ComfyUI exists
if not exist "..\ComfyUI" (
    echo ❌ Error: ComfyUI directory not found at ..\ComfyUI
    echo Please ensure this script is run from ben2-serverless-cpu directory
    exit /b 1
)

echo ✓ ComfyUI directory found
echo.
echo Starting build (this will take 10-15 minutes)...
echo.

REM Build from parent directory to access ComfyUI
cd ..
docker build -f ben2-serverless-cpu\Dockerfile -t %FULL_IMAGE% .

if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ Build failed!
    exit /b 1
)

echo.
echo ======================================
echo ✅ Build Complete!
echo ======================================
echo Image: %FULL_IMAGE%
echo.

REM Show image size
docker images %IMAGE_NAME% --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo.
echo Next steps:
echo 1. Test locally: docker run -it %FULL_IMAGE%
echo 2. Push to registry: docker push your-registry/%FULL_IMAGE%
echo 3. Deploy to RunPod with CPU worker type

endlocal
