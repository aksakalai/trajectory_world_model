@echo off
setlocal EnableExtensions

set "GENERATOR_DIR=%~dp0"
set "GENERATOR_EXE=%GENERATOR_DIR%he_grenade_game.exe"
set "GENERATOR_CONFIG=%GENERATOR_DIR%generator-comparison-config.json"
set "REVIEW_TOOL=%GENERATOR_DIR%..\Scripts\review_dataset.py"

if not exist "%GENERATOR_EXE%" (
  echo Generator executable not found:
  echo %GENERATOR_EXE%
  pause
  exit /b 1
)

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "RUN_ID=%%I"
set "RUN_ROOT=%GENERATOR_DIR%GeneratedData\comparison-%RUN_ID%"

for %%R in (320 384) do (
  for %%S in (movement trajectory throw) do (
    echo.
    echo Generating %%S at %%Rx%%R...
    "%GENERATOR_EXE%" -GenerateDataset -GeneratorConfig="%GENERATOR_CONFIG%" -Stage=%%S -Width=%%R -Height=%%R -Output="%RUN_ROOT%\%%S_%%R" -BuildRevision=curriculum-three-stage-comparison -RenderOffscreen -unattended -nosound -NoSplash -NoVSync -log
    if not exist "%RUN_ROOT%\%%S_%%R\dataset.json" goto :failed

    if exist "%REVIEW_TOOL%" (
      if exist "%GENERATOR_DIR%ffmpeg.exe" (
        python "%REVIEW_TOOL%" "%RUN_ROOT%\%%S_%%R" --ffmpeg="%GENERATOR_DIR%ffmpeg.exe"
      ) else (
        python "%REVIEW_TOOL%" "%RUN_ROOT%\%%S_%%R"
      )
      if errorlevel 1 goto :failed
    )
  )
)

echo.
echo All six matched-seed datasets and review videos are ready:
echo %RUN_ROOT%
pause
exit /b 0

:failed
echo.
echo Comparison generation failed. The completed outputs were left intact:
echo %RUN_ROOT%
pause
exit /b 1
