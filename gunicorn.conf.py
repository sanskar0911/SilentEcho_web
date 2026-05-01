import os

# Render dynamically assigns a port via the PORT environment variable.
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Use standard threads instead of eventlet to support Flask-SocketIO in threading mode
worker_class = "gthread"
threads = 2
workers = 1
