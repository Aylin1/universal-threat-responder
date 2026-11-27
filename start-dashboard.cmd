@echo off
setlocal
cd /d "%~dp0"
echo Dashboard setup can take several minutes or longer on the first run.
echo Please keep this window open. The time required depends on downloads and feed refresh.
set "PYTHON="
set "PYTHON_ARGS="
call :find_python
if not defined PYTHON (
    echo Python 3.11 was not found. Installing it for the current Windows user with winget...
    echo This may take several minutes; approve any requested prompts and keep this window open.
    where winget >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Windows Package Manager ^(winget^) is unavailable. Install Python 3.11 and try again.
        goto :failed
    )
    winget install --id Python.Python.3.11 --exact --scope user --silent
    if errorlevel 1 (
        echo ERROR: Python installation failed or was declined.
        goto :failed
    )
    call :find_python
)
if not defined PYTHON (
    echo ERROR: Python 3.11 could not be located after installation. Restart this launcher and try again.
    goto :failed
)
"%PYTHON%" %PYTHON_ARGS% -m scripts.launch_dashboard
if errorlevel 1 goto :failed
exit /b 0

:find_python
if exist ".dashboard-venv\Scripts\python.exe" (
    set "PYTHON=%CD%\.dashboard-venv\Scripts\python.exe"
    exit /b 0
)
if not exist ".dashboard-venv" if exist ".venv\Scripts\python.exe" (
    set "PYTHON=%CD%\.venv\Scripts\python.exe"
    exit /b 0
)
where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; assert sys.version_info[:2] == (3, 11)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON=python"
        set "PYTHON_ARGS="
        exit /b 0
    )
)
where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON=py"
        set "PYTHON_ARGS=-3.11"
        exit /b 0
    )
)
if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
    set "PYTHON=%LocalAppData%\Programs\Python\Python311\python.exe"
    set "PYTHON_ARGS="
)
exit /b 0

:failed
echo Dashboard startup failed. See the error above.
pause
exit /b 1
