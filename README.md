## Introduction:
An implementation of the board game besieged city with a rust backend and a pyQt GUI, supporting both PVP and PVE.
## Build:
1. Use git to download source code and enter the directory.
```bash
git clone https://github.com/SafetyComputer/Ruspy-City
cd Ruspy-City
```
2. Use uv to download necessary python packages.
```bash
uv sync
```
3. Use maturin to compile rust libraries and convert them to python.packages
```bash
maturin develop --release
```
4. Use pyinstaller to bundle the python application (Optional).
```bash
pyinstaller --onefile game.py
```