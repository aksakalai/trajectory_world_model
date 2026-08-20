@echo off
setlocal

set "GENERATOR_DIR=%~dp0"
set "GENERATOR_EXE=%GENERATOR_DIR%he_grenade_game.exe"
set "GENERATOR_CONFIG=%GENERATOR_DIR%generator-config.json"
set "FINALIZER=%GENERATOR_DIR%..\Scripts\finalize_production_dataset.py"
set "REVIEW_TOOL=%GENERATOR_DIR%..\Scripts\review_dataset.py"
set "OUTPUT_DIR=%GENERATOR_DIR%GeneratedData\small-pilot"

if not exist "%GENERATOR_EXE%" (
  echo Generator executable not found:
  echo %GENERATOR_EXE%
  pause
  exit /b 1
)

if not exist "%FINALIZER%" (
  echo Production finalizer not found:
  echo %FINALIZER%
  pause
  exit /b 1
)

if not exist "%REVIEW_TOOL%" (
  echo Dataset validator not found:
  echo %REVIEW_TOOL%
  pause
  exit /b 1
)

"%GENERATOR_EXE%" -GenerateDataset -GeneratorConfig="%GENERATOR_CONFIG%" -BuildRevision=storage-production-v1 -RenderOffscreen -unattended -nosound -NoSplash -NoVSync -log
set "GENERATOR_EXIT=%ERRORLEVEL%"

if not "%GENERATOR_EXIT%"=="0" (
  echo.
  echo Generation failed with exit code %GENERATOR_EXIT%.
  pause
  exit /b %GENERATOR_EXIT%
)

python "%FINALIZER%" "%OUTPUT_DIR%"
if errorlevel 1 (
  echo.
  echo Parquet finalization failed. The staging files were left intact.
  pause
  exit /b 1
)

python "%REVIEW_TOOL%" "%OUTPUT_DIR%" --validate-only
if errorlevel 1 (
  echo.
  echo Dataset validation failed.
  pause
  exit /b 1
)

echo.
echo Production generation and validation completed. See GeneratedData\small-pilot.
pause
