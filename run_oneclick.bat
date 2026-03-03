@echo off
setlocal enabledelayedexpansion
rem Expect the .venv (Python 3.12) to already be created.
call .venv\Scripts\activate
python oneclick\run_all.py
