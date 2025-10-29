import sys
import subprocess
import yaml
from pathlib import Path

LOG_FILE = "setup_log.txt"
ENV_FILE = "environment.yml"
ENV_NAME = "afml"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
    print(msg)

def validate_python_version(min_major=3, min_minor=10):
    major, minor = sys.version_info[:2]
    if (major, minor) < (min_major, min_minor):
        raise RuntimeError(f"Python {min_major}.{min_minor}+ required, found {major}.{minor}")
    log(f"Python version validated: {major}.{minor}")

def create_env_with_mamba(env_file, env_name):
    log(f"Creating environment '{env_name}' using {env_file}...")
    subprocess.run(["mamba", "env", "create", "-n", env_name, "-f", env_file], check=True)
    log("Environment created successfully.")

def activate_env(env_name):
    activate_cmd = f"conda activate {env_name}"
    log(f"To activate the environment, run:\n{activate_cmd}")

def log_versions(env_name):
    log("Logging Python and package versions...")
    subprocess.run(["conda", "run", "-n", env_name, "python", "--version"], stdout=open(LOG_FILE, "a"))
    subprocess.run(["conda", "run", "-n", env_name, "mamba", "list"], stdout=open(LOG_FILE, "a"))
    subprocess.run(["conda", "run", "-n", env_name, "pip", "freeze"], stdout=open(LOG_FILE, "a"))

def extract_pip_packages(env_file):
    pip_packages = []
    with open(env_file) as f:
        data = yaml.safe_load(f)
        for dep in data.get("dependencies", []):
            if isinstance(dep, dict) and "pip" in dep:
                pip_packages.extend(dep["pip"])
    return pip_packages

def install_pip_packages(env_name, packages):
    if packages:
        log(f"Installing pip-only packages: {packages}")
        subprocess.run(["conda", "run", "-n", env_name, "pip", "install"] + packages, check=True)
        log("Pip packages installed.")

def main():
    try:
        validate_python_version()
        create_env_with_mamba(ENV_FILE, ENV_NAME)
        pip_packages = extract_pip_packages(ENV_FILE)
        install_pip_packages(ENV_NAME, pip_packages)
        log_versions(ENV_NAME)
        activate_env(ENV_NAME)
        log("Environment setup complete.")
    except Exception as e:
        log(f"ERROR: {e}")

if __name__ == "__main__":
    main()
