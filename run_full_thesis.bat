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
echo   1. Full Thesis  (Baseline + HKGAN, 5 datasets x 5 seeds)
echo   2. Ablation Only  (5 datasets x 5 seeds x 4 variants)
echo   3. Cleanup Only  (backup + delete checkpoints)
echo   4. Exit
echo.
set CHOICE=
set /p CHOICE=Select [1-4]:
if "%CHOICE%"=="1" goto FULL_THESIS
if "%CHOICE%"=="2" goto ABLATION
if "%CHOICE%"=="3" goto CLEANUP_ONLY
if "%CHOICE%"=="4" goto END
echo Invalid choice, try again.
timeout /t 1 >nul
goto MENU

:FULL_THESIS
cls
echo [1/2] Cleaning up old checkpoints...
python run_experiments.py --cleanup-only --execute
if errorlevel 1 (
    echo ERROR: cleanup failed, aborting.
    pause
    goto MENU
)
echo.
echo [2/2] Running full thesis (Baseline + HKGAN + Ablation, 5 seeds)...
echo Press Ctrl+C to stop. The menu will reappear after stopping.
echo.
python run_experiments.py --full-thesis --multi-seed --auto-cleanup
echo.
echo Finished (exit code: !ERRORLEVEL!).
pause
goto MENU

:ABLATION
cls
echo [1/2] Cleaning up old checkpoints...
python run_experiments.py --cleanup-only --execute
if errorlevel 1 (
    echo ERROR: cleanup failed, aborting.
    pause
    goto MENU
)
echo.
echo [2/2] Running ablation study (5 datasets x 5 seeds)...
echo Press Ctrl+C to stop. The menu will reappear after stopping.
echo.
python run_ablation.py --full-study --multi-seed --auto-cleanup
echo.
echo Finished (exit code: !ERRORLEVEL!).
pause
goto MENU

:CLEANUP_ONLY
cls
echo Running cleanup...
python run_experiments.py --cleanup-only --execute
echo.
echo Finished (exit code: !ERRORLEVEL!).
pause
goto MENU

:END
endlocal
exit /b 0
