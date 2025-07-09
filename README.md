## Introduction:
An implementation of the board game besieged city with a rust backend and a pyQt GUI, supporting both PVP and PVE.
## Build:
1. Configure build settings.
If cuda is not required, replace the default setting with the following in pyproject.toml to optimize size of the binarry.
```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cpu"},
]
torchvision = [
  { index = "pytorch-cpu"},
]
```
2. Use git to download source code and enter the directory.
```bash
git clone https://github.com/SafetyComputer/Ruspy-City
cd Ruspy-City
.venv/Scripts/activate
```
3. Use uv to download necessary python packages.
```bash
uv sync
```
4. Use maturin to compile rust libraries and convert them to python.packages
```bash
maturin develop --release
```
5. Use pyinstaller to bundle the python application (Optional).
```bash
pyinstaller --onefile game.py
```