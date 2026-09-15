"""Test HTTPS connectivity to DeepSeek API and PyPI (no packages installed).

Uses only Python's built-in urllib. This answers:
  - Can we reach the DeepSeek API endpoint?
  - Is PyPI also unreachable (which would explain pip install failing)?
"""
import urllib.error
import urllib.request

TARGETS = [
    ("DeepSeek API root", "https://api.deepseek.com/"),
    ("DeepSeek chat endpoint", "https://api.deepseek.com/chat/completions"),
    ("PyPI simple index", "https://pypi.org/simple/"),
]

for name, url in TARGETS:
    try:
        req = urllib.request.Request(url, method="GET", headers={"User-Agent": "net-test/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            print(f"[REACHABLE] {name}: HTTP {r.status}")
    except urllib.error.HTTPError as e:
        # An HTTP error (401/404/405) still means we reached the server.
        print(f"[REACHABLE] {name}: HTTP {e.code}")
    except Exception as e:
        print(f"[BLOCKED]   {name}: {type(e).__name__}: {e}")
