from configs.config_default import *

import getpass
import importlib

user = getpass.getuser()

try:
    globals().update(importlib.import_module(f"configs.config_{user}").__dict__)
except ImportError:
    print(f"No config found for {user}. Using defaults.")
