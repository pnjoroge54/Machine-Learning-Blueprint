#!/usr/bin/env python3
import subprocess
from pathlib import Path

import yaml

# === CONFIG ===
ENV_NAME = "afml"  # Replace with your environment name
OUTPUT_FILE = Path("environment.yml")
PINNED_PACKAGES = ["python", "numpy", "pandas"]  # Add more if needed

# === STEP 1: Export environment ===
raw_yaml = subprocess.run(
    ["conda", "env", "export", "--name", ENV_NAME, "--no-builds"], capture_output=True, text=True
).stdout

# === STEP 2: Parse and clean YAML ===
env_data = yaml.safe_load(raw_yaml)
env_data.pop("prefix", None)  # Remove system-specific path


# === STEP 3: Pin key packages ===
def pin_versions(deps, pins):
    for i, dep in enumerate(deps):
        if isinstance(dep, str):
            for pkg in pins:
                if dep.startswith(pkg + "=") or dep == pkg:
                    # Already pinned or needs pinning
                    version = (
                        subprocess.run(
                            ["conda", "list", "-n", ENV_NAME, pkg], capture_output=True, text=True
                        )
                        .stdout.splitlines()[-1]
                        .split()[1]
                    )
                    deps[i] = f"{pkg}={version}"
    return deps


env_data["dependencies"] = pin_versions(env_data["dependencies"], PINNED_PACKAGES)

# === STEP 4: Save cleaned file ===
with OUTPUT_FILE.open("w") as f:
    yaml.dump(env_data, f, sort_keys=False)

print(f"✅ Cleaned environment file saved to {OUTPUT_FILE}")
print(f"✅ Cleaned environment file saved to {OUTPUT_FILE}")
