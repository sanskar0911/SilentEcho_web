import os

# Render dynamically assigns a port via the PORT environment variable.
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# These are fallback settings in case the CLI flags are missing
worker_class = "eventlet"
workers = 1
