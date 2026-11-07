@echo off
cd /d "%~dp0"
echo ResNet20: przycinanie i kompensacja, real i quaternion, seed 2026.
".venv-cuda\Scripts\python.exe" -B -u -m scripts.resnet20_compression --run
:end
pause
