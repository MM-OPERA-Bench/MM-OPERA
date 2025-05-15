# MM-OPERA/evaluation/config_loader.py

import yaml
import os
from pathlib import Path

# Determine the project root dynamically
# Assuming config_loader.py is in MM-OPERA/evaluation/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "evaluation" / "model_config.yaml"

_config = None


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """
    Loads the YAML configuration file.

    Args:
        config_path (Path): Path to the YAML configuration file.

    Returns:
        dict: The loaded configuration.

    Raises:
        FileNotFoundError: If the config file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    global _config
    if _config is None:
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                _config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration file {config_path}: {e}")
            raise
    return _config


def get_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """
    Returns the loaded configuration, loading it if it hasn't been already.
    """
    return _config if _config else load_config(config_path)


def get_api_key(model_name: str) -> str | None:
    """
    Retrieves the API key for a given model, checking environment variables first.
    """
    config = get_config()
    model_conf = config.get("models", {}).get(model_name)
    if not model_conf:
        print(f"Warning: Model '{model_name}' not found in configuration.")
        return None

    provider_name = model_conf.get("provider")
    if not provider_name:
        print(f"Warning: Provider not specified for model '{model_name}'.")
        return None

    provider_conf = config.get("api_providers", {}).get(provider_name)
    if not provider_conf:
        print(
            f"Warning: Provider '{provider_name}' not found in api_providers configuration."
        )
        return None

    api_key_env_var = provider_conf.get("api_key_env_var")
    if api_key_env_var:
        api_key = os.getenv(api_key_env_var)
        if api_key:
            return api_key
        else:
            print(
                f"Warning: Environment variable '{api_key_env_var}' for model '{model_name}' is not set."
            )
            # Fallback to direct key if it exists (less secure)
            if "api_key" in provider_conf:
                print(
                    f"Warning: Using API key directly from config for provider '{provider_name}'. This is less secure."
                )
                return provider_conf["api_key"]
            return None
    elif "api_key" in provider_conf:  # Direct key (less secure)
        print(
            f"Warning: Using API key directly from config for provider '{provider_name}'. This is less secure."
        )
        return provider_conf["api_key"]

    print(f"Warning: API key configuration not found for provider '{provider_name}'.")
    return None


if __name__ == "__main__":
    # Example usage:
    try:
        cfg = get_config()
        print("Config loaded successfully!")
        print(
            f"GPT-4o API Key (from env if AIGPTX_API_KEY is set): {get_api_key('gpt-4o')}"
        )
        print(
            f"RIA Prompt: {cfg['evaluation_settings']['ria']['prompt'][:50]}..."
        )  # Print first 50 chars
    except Exception as e:
        print(f"Error in config_loader example: {e}")
