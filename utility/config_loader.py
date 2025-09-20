# config_loader.py
import os
from configparser import ConfigParser

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.ini'))
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    config = ConfigParser()
    config.read(config_path)
    return config
