from pathlib import Path
from google.colab import drive, userdata
import sys
import importlib

# Define the GitHubManager class before its usage
class GitHubManager:
    def __init__(self, drive_path: Path, github_user: str, email: str, repo_name: str, branch: str = "main"):
        self.drive_path = drive_path
        self.github_user = github_user
        self.email = email
        self.repo_name = repo_name
        self.branch = branch
        self.repo_path = self.drive_path / self.repo_name
        self.token = None # Will be set by user_data or passed directly

        # Ensure drive is mounted
        self._mount_drive()

    def _mount_drive(self):
        try:
            drive.mount('/content/drive')
            print('Drive mounted successfully.')
        except Exception as e:
            print(f"Error mounting drive: {e}. Attempting forced remount.")
            drive.mount("/content/drive", force_remount=True)
            print('Drive forcibly remounted.')

    def _get_token(self):
        try:
            self.token = userdata.get('GH_TOKEN')
            if self.token:
                print('GitHub token set from userdata.')
            else:
                print('No GitHub token found in userdata.')
        except Exception:
            print('Could not retrieve GitHub token from userdata.')
            self.token = None

    def _get_repo_url(self):
        if self.token:
            return f"https://{self.token}@github.com/{self.github_user}/{self.repo_name}.git"
        else:
            return f"https://github.com/{self.github_user}/{self.repo_name}.git"

    def clone_or_pull_repo(self):
        self._get_token()
        repo_url = self._get_repo_url()

        if self.repo_path.exists():
            print(f"Repository already exists at {self.repo_path}. Pulling latest updates...")
            %cd {self.repo_path}
            !git stash  # Hide any accidental tweaks
            !git pull origin {self.branch}
            # !git stash pop # Uncomment this if you want your tweaks back automatically
        else:
            print(f"Repository not found. Cloning into {self.repo_path}...")
            %cd {self.drive_path}
            !git clone --branch {self.branch} {repo_url}
            %cd {self.repo_path} # Change into the cloned repo directory

        print(f"Working Dir: {Path.cwd()}")

    def add_to_python_path(self):
        if str(self.repo_path) not in sys.path:
            sys.path.append(str(self.repo_path))
            print(f"Added {self.repo_path} to Python path.")
        else:
            print(f"{self.repo_path} already in Python path.")

    def setup_git_credentials(self):
        print("Setting up Git credentials...")
        !git config --global user.email {self.email}
        !git config --global user.name {self.github_user}
        # Set the remote URL including the token
        repo_url = self._get_repo_url()
        !git remote set-url origin {repo_url}
        print("Git credentials set.")

    def configure_git_hooks(self):
        print("Configuring Git hooks...")
        # Give permission to run executables
        !chmod +x .git/hooks/pre-push
        !chmod +x .git/hooks/post-merge
        !chmod +x .git/hooks/post-commit
        print("Git hooks configured.")

    def git_pull(self, branch: str = None, force_remove_lock: bool = True):
        current_branch = branch if branch else self.branch
        print(f"Performing git pull on branch {current_branch}...")
        %cd {self.repo_path}

        if force_remove_lock:
            lock_file = self.repo_path / ".git" / "index.lock"
            if lock_file.exists():
                print("Forcefully removing git index.lock file.")
                !rm -f {lock_file}

        !git pull origin {current_branch}
        print(f"Git pull on {current_branch} complete.")

    def git_hard_reset(self, branch: str = None):
        current_branch = branch if branch else self.branch
        print(f"Performing git hard reset to origin/{current_branch}...")
        %cd {self.repo_path}

        # 1. Skip the broken large files (LFS)
        !git config filter.lfs.smudge "git-lfs smudge --skip -- %f"
        !git config filter.lfs.process "git-lfs filter-process --skip"

        # 2. Fetch the latest data from GitHub without merging yet
        !git fetch origin {current_branch}

        # 3. FORCE your local files to match the branch exactly
        # WARNING: This deletes any unsaved changes in your Colab files
        !git reset --hard origin/{current_branch}

        # 4. Clean up LFS settings
        !git config --unset filter.lfs.smudge
        !git config --unset filter.lfs.process

        print(f"Success! Your local code now perfectly matches the GitHub {current_branch} branch.")

    def git_add_commit_push(self, commit_message: str):
        print("Staging, committing, and pushing changes...")
        %cd {self.repo_path}
        !git add .
        !git commit -m "{commit_message}"
        # Use || for conditional execution: if first fails, try second (for new repos)
        !git push origin {self.branch} || !git remote add origin {self._get_repo_url()} && git push -u origin {self.branch}
        print("Changes pushed to GitHub.")

    @staticmethod
    def reload_module(module_name):
        """A Python 3.12 safe way to reload a specific module."""
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
            print(f"Successfully reloaded {module_name}")
        else:
            print(f"Module {module_name} not found in memory.")

    def full_setup(self):
        print("Starting full GitHub repository setup...")
        self.clone_or_pull_repo()
        self.add_to_python_path()
        self.setup_git_credentials()
        self.configure_git_hooks()
        print("Full GitHub repository setup complete.")

# === Colab Setup ===
drive.mount('/content/drive')
drive_path = Path("/content/drive/MyDrive")

# === CONFIGURATION ===
GITHUB_USER = userdata.get('GH_USERNAME')
EMAIL = userdata.get('GH_EMAIL')
REPO_NAME = "Machine-Learning-Blueprint"
BRANCH = "streamlined"

# Instantiate and run setup
gh_manager = GitHubManager(
    drive_path=drive_path,
    github_user=GITHUB_USER,
    email=EMAIL,
    repo_name=REPO_NAME,
    branch=BRANCH
)
gh_manager.full_setup()

print("Setup complete. You can now run your imports.")