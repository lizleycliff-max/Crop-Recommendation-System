"""
Package app.py (Streamlit) into one standalone executable in ./dist.

Run on the TARGET OS (PyInstaller cannot cross-compile): build on Windows for a .exe.
    pip install pyinstaller streamlit
    python build_desktop.py

Streamlit cannot be frozen by pointing PyInstaller at app.py directly, so this script
generates a small launcher that starts Streamlit's server on the bundled app.py.
"""
import os
import sys
import shutil
import platform
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
APP_NAME = "CropRecommendation"
LAUNCHER = os.path.join(ROOT, "desktop_launcher.py")

LAUNCHER_SRC = '''\
import os, sys, multiprocessing

def resource_path(rel):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from streamlit.web import bootstrap
    app_path = resource_path("app.py")
    os.chdir(os.path.dirname(app_path))
    flags = {
        "server.headless": False,          # opens the default browser
        "server.port": 8501,
        "global.developmentMode": False,
        "browser.gatherUsageStats": False,
        "server.fileWatcherType": "none",
        "theme.primaryColor": "#3F7D58",
        "theme.backgroundColor": "#ffffff",
        "theme.secondaryBackgroundColor": "#F7F8F4",
        "theme.textColor": "#24352B",
    }
    bootstrap.load_config_options(flag_options=flags)
    bootstrap.run(app_path, False, [], flags)
'''


def main():
    if platform.system() != "Windows":
        print(f"WARNING: running on {platform.system()}; this will produce a {platform.system()} "
              "binary, not a .exe. Run on Windows for CropAdvisor.exe.")

    for need in ("app.py", "src", "models"):
        if not os.path.exists(os.path.join(ROOT, need)):
            sys.exit(f"Missing required path: {need}")

    with open(LAUNCHER, "w", encoding="utf-8") as f:
        f.write(LAUNCHER_SRC)

    sep = os.pathsep  # ';' on Windows, ':' elsewhere
    data = [("app.py", "."), ("src", "src"), ("models", "models")]
    if os.path.isdir(os.path.join(ROOT, "data")):
        data.append(("data", "data"))

    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile",
           "--name", APP_NAME, "--distpath", os.path.join(ROOT, "dist"),
           "--workpath", os.path.join(ROOT, "build"), "--specpath", ROOT]
    for src, dst in data:
        cmd += ["--add-data", f"{os.path.join(ROOT, src)}{sep}{dst}"]
    for pkg in ("streamlit", "plotly", "shap", "sklearn", "xgboost", "catboost"):
        cmd += ["--collect-all", pkg]
    for pkg in ("streamlit", "torch", "pandas", "numpy"):
        cmd += ["--copy-metadata", pkg]
    cmd += ["--collect-submodules", "src", "--hidden-import", "joblib", "--hidden-import", "torch",
            LAUNCHER]

    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        sys.exit(f"PyInstaller failed with exit code {rc}")

    exe = os.path.join(ROOT, "dist", APP_NAME + (".exe" if platform.system() == "Windows" else ""))
    print(f"\nBuild complete: {exe}" if os.path.exists(exe) else "\nBuild finished but output not found.")
    shutil.rmtree(os.path.join(ROOT, "build"), ignore_errors=True)


if __name__ == "__main__":
    main()
