@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Cannot activate .venv - check that the virtual environment exists.
    pause
    exit /b 1
)

:MENU
cls
echo ========================================
echo   Thesis Experiment Runner
echo ========================================
echo.
echo   1. Setup Data    (download datasets from HuggingFace, run DAPT)
echo   2. Full Thesis   (Baseline + HKGAN, 5 datasets x 5 seeds)
echo   3. Ablation Only (5 datasets x 5 seeds x 4 variants)
echo   4. Cleanup Only  (backup + delete checkpoints)
echo   5. Exit
echo.
set CHOICE=
set /p CHOICE=Select [1-5]:
if "%CHOICE%"=="1" goto SETUP_DATA
if "%CHOICE%"=="2" goto FULL_THESIS
if "%CHOICE%"=="3" goto ABLATION
if "%CHOICE%"=="4" goto CLEANUP_ONLY
if "%CHOICE%"=="5" goto END
echo Invalid choice, try again.
timeout /t 1 >nul
goto MENU

:SETUP_DATA
cls
echo Setting up data directory...
echo (Download from HuggingFace + optionally run DAPT)
echo.
echo Options:
echo   [1] Full setup (download + DAPT if unlabeled data exists)
echo   [2] Download only (skip DAPT, use pre-trained models from HuggingFace)
echo.
set SETUP_CHOICE=
set /p SETUP_CHOICE=Select [1-2]:
if "%SETUP_CHOICE%"=="1" (
    python 01_setup_data.py
) else (
    python 01_setup_data.py --skip-dapt
)
echo.
echo Finished (exit code: !ERRORLEVEL!).
pause
goto MENU

:FULL_THESIS
cls
echo [1/2] Cleaning up old checkpoints...
python 02_run_experiments.py --cleanup-only --execute
if errorlevel 1 (
    echo ERROR: cleanup failed, aborting.
    pause
    goto MENU
)
echo.
echo [2/2] Running full thesis (Baseline + HKGAN + Ablation, 5 seeds)...
echo Press Ctrl+C to stop. The menu will reappear after stopping.
echo.
python 02_run_experiments.py --full-thesis --multi-seed --auto-cleanup
echo.
echo Finished (exit code: !ERRORLEVEL!).
pause
goto MENU

:ABLATION
cls
echo [1/2] Cleaning up old checkpoints...
python 02_run_experiments.py --cleanup-only --execute
if errorlevel 1 (
    echo ERROR: cleanup failed, aborting.
    pause
    goto MENU
)
echo.
echo [2/2] Running ablation study (5 datasets x 5 seeds)...
echo Press Ctrl+C to stop. The menu will reappear after stopping.
echo.
python 03_run_ablation.py --full-study --multi-seed --auto-cleanup
echo.
echo Finished (exit code: !ERRORLEVEL!).
pause
goto MENU

:CLEANUP_ONLY
cls
echo Running cleanup...
python 02_run_experiments.py --cleanup-only --execute
echo.
echo Finished (exit code: !ERRORLEVEL!).
pause
goto MENU

:END
endlocal
exit /b 0
