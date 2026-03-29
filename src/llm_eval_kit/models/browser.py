"""
Browser-based Model Provider — automates a web-based LLM chat UI via Playwright.

=== WHAT IS THIS? ===

Some LLMs are only accessible through a web interface (no API).
For example:
    - Open WebUI (local LLM frontend)
    - HuggingChat
    - Any custom LLM chat deployment

This provider launches a browser, types the prompt into the chat input,
clicks send, waits for the response to appear, and extracts the text.
It's like a human using the chat — but automated.

=== WHY IS THIS USEFUL? ===

1. Evaluate LLMs that don't have an API
2. Test LLM web interfaces themselves (is the UI working?)
3. Compare API responses vs web UI responses

=== NEW PLAYWRIGHT CONCEPTS ===

- Persistent browser contexts (reuse sessions/cookies across calls)
- page.wait_for_selector() — wait for dynamic elements to appear
- page.wait_for_function() — wait for custom JS conditions
- page.type() vs page.fill() — keystroke simulation vs instant fill
- Handling streaming responses (text that appears gradually)
- Retry logic with Playwright timeouts
- Browser state management (keeping logged in)
"""

from dataclasses import dataclass, field
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page

from llm_eval_kit.models.base import BaseLLM


@dataclass
class BrowserChatConfig:
    """Configuration for automating a web-based LLM chat interface.

    Each chat UI is different, so the user must provide CSS selectors
    that match THEIR specific chat interface. These defaults work with
    Open WebUI (https://github.com/open-webui/open-webui), a popular
    local LLM frontend.

    === HOW TO FIND THESE SELECTORS ===
    1. Open the chat UI in Chrome
    2. Right-click on the input box → "Inspect"
    3. Look at the HTML element and find a unique selector
    4. Do the same for the send button and response area
    """
    # URL of the chat interface
    url: str = "http://localhost:3000"

    # CSS selector for the text input where you type your prompt
    input_selector: str = "textarea#chat-input"

    # CSS selector for the send/submit button
    send_selector: str = "button[type='submit']"

    # CSS selector for the latest response message
    # The scraper grabs the LAST element matching this selector
    response_selector: str = ".message.assistant .content"

    # How long to wait for a response (ms) — LLMs can be slow!
    response_timeout_ms: int = 120000

    # Whether to run the browser in headless mode
    # Set to False during development to WATCH what's happening
    headless: bool = True

    # Whether to type character by character (True) or fill instantly (False)
    # Character-by-character is slower but more realistic — some chat UIs
    # have per-keystroke handlers that break with instant fill.
    simulate_typing: bool = False

    # Delay between keystrokes when simulate_typing=True (ms)
    typing_delay_ms: int = 30


class BrowserModel(BaseLLM):
    """A model provider that uses Playwright to interact with a web chat UI.

    === PLAYWRIGHT CONCEPT: Persistent state across calls ===

    Unlike our scraper (which opens/closes the browser for each scrape),
    this model keeps the browser OPEN between generate() calls.

    Why? Because:
    1. Starting a browser is slow (~500ms). Opening one per prompt is wasteful.
    2. The chat UI might require login — we want to stay logged in.
    3. Some chat UIs maintain conversation context — closing resets it.

    The browser is opened lazily (on first generate() call) and closed
    when close() is called or the object is garbage collected.
    """

    def __init__(self, config: BrowserChatConfig | None = None):
        self.config = config or BrowserChatConfig()
        self.name = f"browser/{self.config.url}"

        # These are initialized lazily in _ensure_browser()
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    def _ensure_browser(self) -> Page:
        """Launch the browser if not already running, and return the page.

        === PLAYWRIGHT CONCEPT: Lazy initialization ===

        We don't launch the browser in __init__ because:
        1. The user might create the model but never use it
        2. We want to control WHEN the browser starts
        3. The runner might be in dry-run mode (no browser needed)

        === PLAYWRIGHT CONCEPT: Context reuse ===

        A BrowserContext is like an incognito session — it has its own:
        - Cookies
        - LocalStorage
        - SessionStorage
        - Cache

        By reusing the SAME context for all generate() calls,
        we keep the user's session (login state, etc.) alive.
        """
        if self._page is not None:
            return self._page

        self._playwright = sync_playwright().start()

        self._browser = self._playwright.chromium.launch(
            headless=self.config.headless,
        )

        # Create a context — this is our persistent "session"
        self._context = self._browser.new_context(
            viewport={"width": 1280, "height": 720},
        )

        self._page = self._context.new_page()

        # Navigate to the chat UI
        self._page.goto(self.config.url, wait_until="networkidle")

        return self._page

    def generate(self, prompt: str) -> str:
        """Type a prompt into the chat UI and extract the response.

        === STEP-BY-STEP FLOW ===

        1. Ensure browser is running and on the chat page
        2. Count existing response messages (to know when a NEW one appears)
        3. Type the prompt into the input field
        4. Click send
        5. Wait for a NEW response message to appear
        6. Wait for the response to stop changing (streaming complete)
        7. Extract and return the response text
        """
        page = self._ensure_browser()

        # ── Step 1: Count existing responses ──
        # We need to know how many response messages exist BEFORE we send,
        # so we can detect when a NEW one appears.
        existing_count = page.locator(self.config.response_selector).count()

        # ── Step 2: Type the prompt ──
        #
        # === PLAYWRIGHT CONCEPT: fill() vs type() ===
        #
        # fill(): Sets the input value INSTANTLY. Like copy-paste.
        #   - Fast (good for tests)
        #   - But some JS frameworks don't detect the change
        #     (React's onChange might not fire)
        #
        # type(): Simulates individual keystrokes. Key down, key up, for each char.
        #   - Slow (types one char at a time)
        #   - But triggers ALL keyboard events (keydown, keypress, keyup, input)
        #   - More realistic — works with ALL chat UIs
        #
        # We let the user choose via config.simulate_typing

        input_field = page.locator(self.config.input_selector)
        input_field.click()  # Focus the input first

        if self.config.simulate_typing:
            # type() simulates real keystrokes with a delay between each
            input_field.type(prompt, delay=self.config.typing_delay_ms)
        else:
            # fill() is instant — good enough for most UIs
            input_field.fill(prompt)

        # ── Step 3: Click send ──
        page.locator(self.config.send_selector).click()

        # ── Step 4: Wait for new response ──
        #
        # === PLAYWRIGHT CONCEPT: wait_for_function() ===
        #
        # Runs a JavaScript function repeatedly until it returns true.
        # This is the MOST FLEXIBLE way to wait for a condition.
        #
        # We wait for the number of response elements to increase,
        # meaning the LLM has started generating a response.
        #
        # The function receives `expected_count` as an argument
        # (passed via `arg=`).

        expected_count = existing_count + 1

        page.wait_for_function(
            """(args) => {
                const [selector, count] = args;
                return document.querySelectorAll(selector).length >= count;
            }""",
            arg=[self.config.response_selector, expected_count],
            timeout=self.config.response_timeout_ms,
        )

        # ── Step 5: Wait for streaming to complete ──
        #
        # LLM responses appear word-by-word (streaming). We need to wait
        # until the response STOPS CHANGING before extracting it.
        #
        # Strategy: Check the text, wait a bit, check again. If it's the
        # same, streaming is done.
        #
        # === PLAYWRIGHT CONCEPT: page.evaluate() for complex checks ===

        response_text = self._wait_for_stable_response(page, expected_count - 1)

        return response_text

    def _wait_for_stable_response(
        self,
        page: Page,
        response_index: int,
        stability_checks: int = 3,
        check_interval_ms: int = 500,
    ) -> str:
        """Wait for the response to stop changing (streaming complete).

        We read the response text multiple times with a delay. When it
        stays the same for `stability_checks` consecutive reads, we
        consider it stable (streaming is done).

        This is a common pattern for handling streaming/progressive content
        in Playwright tests.
        """
        last_text = ""
        stable_count = 0

        for _ in range(60):  # Max 30 seconds (60 * 500ms)
            current_text = page.evaluate(
                """(args) => {
                    const [selector, index] = args;
                    const elements = document.querySelectorAll(selector);
                    return elements[index]?.innerText || '';
                }""",
                arg=[self.config.response_selector, response_index],
            )

            if current_text == last_text and current_text:
                stable_count += 1
                if stable_count >= stability_checks:
                    return current_text.strip()
            else:
                stable_count = 0
                last_text = current_text

            page.wait_for_timeout(check_interval_ms)

        # Timeout — return whatever we have
        return last_text.strip() if last_text else "[browser-model: no response detected]"

    def close(self):
        """Shut down the browser and clean up resources.

        === PLAYWRIGHT CONCEPT: Cleanup order matters ===

        You MUST close in reverse order:
            page → context → browser → playwright

        If you close the browser before the context, you get errors.
        """
        if self._page:
            self._page.close()
            self._page = None
        if self._context:
            self._context.close()
            self._context = None
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None

    def __del__(self):
        """Cleanup when the object is garbage collected."""
        self.close()
