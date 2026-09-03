@echo off
chcp 65001 >nul
title WM-VLA 演示启动器
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-demo.ps1"
if errorlevel 1 (
  echo.
  echo 启动失败，请查看上方错误或 .runtime\dev.stderr.log
  pause
)
