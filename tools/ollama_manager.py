# ai/tools/ollama_manager.py
import subprocess
import shutil
from typing import Optional, Literal
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaManager:
    """
    Handles Ollama model operations including:
    - Checking model availability
    - Pulling models
    - Validating local installations
    """

    def __init__(self, ollama_path: Optional[str] = None):
        self.ollama_bin = self._locate_ollama(ollama_path)
        self.required_models = {
            'llm': 'mistral',
            'embeddings': 'nomic-embed-text'
        }

    def _locate_ollama(self, custom_path: Optional[str]) -> str:
        """Find Ollama executable path"""
        if custom_path and Path(custom_path).exists():
            return custom_path
        
        # Check common installation paths
        default_paths = [
            '/usr/local/bin/ollama',
            str(Path.home() / '.ollama/bin/ollama'),
            'ollama'  # Try system PATH
        ]
        
        for path in default_paths:
            if shutil.which(path):
                return path
        
        raise FileNotFoundError(
            "Ollama not found. Install from https://ollama.ai/"
        )

    def _run_command(self, cmd: str) -> str:
        """Execute shell command with error handling"""
        try:
            result = subprocess.run(
                cmd.split(),
                check=True,
                text=True,
                capture_output=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e.stderr}")
            raise

    def model_exists(self, model_name: str) -> bool:
        """Check if model is available locally"""
        try:
            models = self._run_command(f"{self.ollama_bin} list")
            return model_name in models
        except Exception:
            return False

    def pull_model(self, model_name: str, quiet: bool = False) -> None:
        """
        Pull a model from Ollama registry
        Args:
            model_name: Name of model (e.g. 'mistral')
            quiet: If True, suppresses output
        """
        if self.model_exists(model_name):
            logger.info(f"Model '{model_name}' already exists")
            return

        logger.info(f"Downloading model '{model_name}'...")
        cmd = f"{self.ollama_bin} pull {model_name}"
        if quiet:
            cmd += " > /dev/null 2>&1"
        self._run_command(cmd)
        logger.info(f"Model '{model_name}' ready")

    def verify_models(self) -> None:
        """Ensure all required models are available"""
        missing = []
        for model_type, model_name in self.required_models.items():
            if not self.model_exists(model_name):
                missing.append((model_type, model_name))
        
        if missing:
            logger.warning("Missing required models:")
            for model_type, model_name in missing:
                logger.warning(f"- {model_type}: {model_name}")
            self._download_missing(missing)

    def _download_missing(self, models: list) -> None:
        """Download missing models with progress"""
        from tqdm import tqdm
        for model_type, model_name in tqdm(models, desc="Downloading models"):
            self.pull_model(model_name, quiet=True)

    def get_model_path(self, model_name: str) -> Path:
        """Get local path to model files"""
        ollama_home = Path.home() / '.ollama'
        model_file = ollama_home / 'models' / 'manifests' / 'registry.ollama.ai' / model_name
        if not model_file.exists():
            raise FileNotFoundError(f"Model files not found at {model_file}")
        return model_file

# Test cases
if __name__ == "__main__":
    manager = OllamaManager()
    
    # Verify core models
    print("Verifying models...")
    manager.verify_models()
    
    # Test model paths
    try:
        path = manager.get_model_path("mistral")
        print(f"Mistral model path: {path}")
    except Exception as e:
        print(f"Error: {e}")