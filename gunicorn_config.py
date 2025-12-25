bind = "0.0.0.0:8080"
workers = 9
threads = 2 
timeout = 300  # Increase the timeout (in seconds) to prevent timeouts
keepalive = 5 
preload_app = True  # Preload the app to save memory and improve performance
worker_class = "gthread"  # Use threaded workers to optimize for I/O-bound tasks