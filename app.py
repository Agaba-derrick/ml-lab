"""
Root application entrypoint for Hugging Face Spaces & cloud deployments.
Exposes `app` directly for Gunicorn / WSGI and python execution.
"""
import importlib
import os

iris_module = importlib.import_module("01_iris_project.app")
app = iris_module.app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False)
