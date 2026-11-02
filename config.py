"""
Runtime configuration, read from environment variables.

Nothing is hard-coded: every key, URL and model name comes from the
environment, so the same code runs against any OpenAI-compatible LLM
endpoint and with or without a Semantic Scholar API key.
"""

import os

# Semantic Scholar (https://www.semanticscholar.org/product/api).
# The key is optional; it raises the rate limit.
SS_API_KEY = os.environ.get("SS_API_KEY", "")
SS_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
# Minimum seconds between Semantic Scholar calls, shared by all threads.
SS_MIN_INTERVAL = float(os.environ.get("SS_MIN_INTERVAL", "1.0"))

# Any OpenAI-compatible chat-completions endpoint, used for extraction.
LLM_URL = os.environ.get("LLM_URL", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "60"))
