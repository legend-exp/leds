---
name: run-app
description:
  Launch the leds Panel app against the local mock data and drive/screenshot it
  with headless Chrome (Playwright) to verify a change visually
---

# Run the leds app and look at it

## Launch the server

The venv install is **non-editable** — reinstall after source edits or the
server runs stale code. Use `--reinstall --no-cache`: uv otherwise reuses a
cached wheel when the setuptools_scm version hasn't changed (same day, same
commit) and silently installs the OLD code:

```bash
cd /Users/georgemarshall/Desktop/Legend/code/leds
~/.local/bin/uv pip install -p .venv --no-deps --reinstall --no-cache -q .
lsof -ti:5006 -sTCP:LISTEN | xargs -r kill   # free the port from a previous run
LEDS_BASE_PATH=/Users/georgemarshall/mock_prod nohup .venv/bin/leds serve \
    --port 5006 --allow-websocket-origin localhost:5006 > /tmp/leds-serve.log 2>&1 &
# macOS has no `timeout`; poll the port
for i in $(seq 1 60); do curl -sf http://localhost:5006 >/dev/null && break; sleep 1; done
```

Mock data: period p19, run r001, one 1 h file (~8046 events). Stop the server
with the same `lsof ... | xargs -r kill` line.

## Drive it headless

No chromium-cli on this machine; use Playwright with the **system Chrome**
(`channel="chrome"`, no browser download). One-time setup:

```bash
~/.local/bin/uv venv -q /tmp/pwvenv --python 3.11
~/.local/bin/uv pip install -q -p /tmp/pwvenv playwright
```

Driver skeleton (`/tmp/pwvenv/bin/python drive.py`):

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1600, "height": 1100})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("http://localhost:5006")
    page.wait_for_selector("text=Validation", timeout=60000)
    page.wait_for_timeout(3000)  # first event render
    page.click("text=Validation")  # tab names are plain text targets
    page.wait_for_timeout(4000)  # tabs build lazily on activation
    page.screenshot(path="/tmp/leds.png", full_page=True)
    print("JS errors:", errors or "none")
    browser.close()
```

**Gotchas (all hit for real):**

- Bokeh 3 renders widgets inside **shadow DOM**: XPath selectors never match.
  Playwright _CSS_ selectors pierce shadow roots — find a Panel `Select` by one
  of its options, then `select_option`:
  ```python
  show = page.locator("select:has(option[value='trigger rates'])")
  show.select_option("calibration summary")
  page.wait_for_timeout(3000)
  ```
- Panel `Checkbox` widgets: clicking the label text does NOT toggle them, and
  `label:has-text(...) input` doesn't match either (the input is a direct
  shadow-root child). Enumerate `page.locator("input[type=checkbox]")` and pick
  by index (sidebar order first, then tab controls), then `.click(force=True)`
  and assert `.is_checked()` flipped.
- Tab content builds on first activation (the updaters gate on the active tab;
  the tabs themselves are static) — wait after clicking a tab or changing a
  control, then screenshot.
- Always print the collected `pageerror`s before declaring success.
- **Look at the screenshot** (Read the PNG); a blank frame means the websocket
  origin is wrong (`--allow-websocket-origin localhost:<port>`).
