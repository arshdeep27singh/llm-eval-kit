"""
Tests for the browser-based model provider.

=== THE CHALLENGE ===

How do you test a model provider that automates a web chat UI
without having a REAL chat UI running?

Answer: We build a FAKE one! We create a minimal HTML page that
mimics a chat interface, serve it locally, and test against that.

=== NEW PLAYWRIGHT CONCEPTS ===

- Starting a local HTTP server for testing
- page.route() to mock API responses
- page.on("console") to capture browser console logs
- page.wait_for_event() to wait for specific events
- Testing against locally-served pages
"""

import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest

from llm_eval_kit.models.browser import BrowserChatConfig, BrowserModel


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES — Fake chat UI and local server
# ══════════════════════════════════════════════════════════════════════════════


FAKE_CHAT_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Fake Chat UI</title>
    <style>
        body { font-family: sans-serif; padding: 20px; }
        #messages { border: 1px solid #ccc; padding: 10px; min-height: 200px;
                    margin-bottom: 10px; max-height: 400px; overflow-y: auto; }
        .message { margin: 8px 0; padding: 8px; border-radius: 8px; }
        .message.user { background: #e3f2fd; }
        .message.assistant .content { background: #f5f5f5; padding: 8px; border-radius: 8px; }
        form { display: flex; gap: 8px; }
        textarea { flex: 1; padding: 8px; resize: none; }
        button { padding: 8px 16px; cursor: pointer; }
    </style>
</head>
<body>
    <h2>🤖 Fake Chat</h2>
    <div id="messages"></div>
    <form id="chatForm">
        <textarea id="chat-input" rows="2" placeholder="Type a message..."></textarea>
        <button type="submit">Send</button>
    </form>

    <script>
        /*
         * This is a FAKE LLM chat UI for testing.
         *
         * When the user sends a message, it:
         * 1. Shows the user's message immediately
         * 2. Waits 300ms (simulating LLM "thinking")
         * 3. Shows a canned response based on the prompt
         * 4. Simulates streaming by revealing the response word-by-word
         *
         * The canned responses let our tests verify exact output.
         */

        const CANNED_RESPONSES = {
            "what is python": "Python is a high-level programming language",
            "what is 2+2": "The answer is 4",
            "hello": "Hello! How can I help you today?",
        };
        const DEFAULT_RESPONSE = "I don't understand that question.";

        const messagesDiv = document.getElementById('messages');
        const form = document.getElementById('chatForm');
        const input = document.getElementById('chat-input');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const prompt = input.value.trim();
            if (!prompt) return;

            // Show user message
            const userMsg = document.createElement('div');
            userMsg.className = 'message user';
            userMsg.textContent = prompt;
            messagesDiv.appendChild(userMsg);

            // Clear input
            input.value = '';

            // Look up canned response
            const key = prompt.toLowerCase().trim();
            const response = CANNED_RESPONSES[key] || DEFAULT_RESPONSE;

            // Simulate delay (LLM "thinking")
            await new Promise(r => setTimeout(r, 300));

            // Create assistant message container
            const assistantMsg = document.createElement('div');
            assistantMsg.className = 'message assistant';
            const content = document.createElement('div');
            content.className = 'content';
            assistantMsg.appendChild(content);
            messagesDiv.appendChild(assistantMsg);

            // Simulate streaming: add words one at a time
            const words = response.split(' ');
            for (let i = 0; i < words.length; i++) {
                await new Promise(r => setTimeout(r, 50));
                content.textContent = words.slice(0, i + 1).join(' ');
            }

            // Scroll to bottom
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        });
    </script>
</body>
</html>"""


class FakeChatServer:
    """A simple HTTP server that serves our fake chat HTML.

    === WHY A SERVER (not file://) ===

    Some web features don't work with file:// protocol:
    - form.addEventListener("submit") might not fire
    - fetch() calls are blocked (CORS)
    - Some JS APIs require HTTP context

    So we serve the HTML via a real HTTP server on localhost.

    === PLAYWRIGHT CONCEPT: Testing against local servers ===

    This is a VERY common pattern:
    1. Start a local server in a background thread
    2. Point Playwright at http://localhost:PORT
    3. Run your tests
    4. Shut down the server

    The server runs in a DAEMON thread — it dies when the test process exits.
    """

    def __init__(self, html_content: str, port: int = 0):
        self.html_content = html_content
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port = port

    def start(self) -> str:
        """Start the server and return its URL."""
        parent = self

        class Handler(SimpleHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(parent.html_content.encode())

            def log_message(self, format, *args):
                pass  # Suppress server logs during tests

        self._server = HTTPServer(("127.0.0.1", self.port), Handler)
        self.port = self._server.server_address[1]

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        return f"http://127.0.0.1:{self.port}"

    def stop(self):
        if self._server:
            self._server.shutdown()


@pytest.fixture
def chat_server():
    """Start a fake chat server for the duration of the test."""
    server = FakeChatServer(FAKE_CHAT_HTML)
    url = server.start()
    yield url
    server.stop()


@pytest.fixture
def browser_config(chat_server: str) -> BrowserChatConfig:
    """Create a BrowserChatConfig pointing at our fake chat."""
    return BrowserChatConfig(
        url=chat_server,
        input_selector="textarea#chat-input",
        send_selector="button[type='submit']",
        response_selector=".message.assistant .content",
        response_timeout_ms=10000,
        headless=True,
        simulate_typing=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 1: Basic Browser Model Functionality
# ══════════════════════════════════════════════════════════════════════════════

class TestBrowserModelBasic:
    """Test basic prompt/response flow through the browser model."""

    def test_simple_prompt_response(self, browser_config: BrowserChatConfig):
        """
        === FULL LIFECYCLE TEST ===

        This tests the complete flow:
        1. BrowserModel creates (but doesn't launch browser yet — lazy init)
        2. generate() launches browser, navigates to chat UI
        3. Types "what is python" into the input
        4. Clicks send
        5. Waits for the response to appear
        6. Returns the response text
        7. close() shuts everything down

        The fake chat UI responds with canned text, so we can assert exactly.
        """
        model = BrowserModel(config=browser_config)
        try:
            response = model.generate("what is python")
            assert response == "Python is a high-level programming language"
        finally:
            model.close()

    def test_model_name(self, browser_config: BrowserChatConfig):
        """The model name should include the URL."""
        model = BrowserModel(config=browser_config)
        assert "browser/" in model.name
        assert "127.0.0.1" in model.name

    def test_multiple_prompts_same_session(self, browser_config: BrowserChatConfig):
        """
        === PLAYWRIGHT CONCEPT: Session persistence ===

        The browser stays open between generate() calls.
        Each call adds to the chat history (like a real conversation).
        The model tracks response count to extract only the LATEST one.
        """
        model = BrowserModel(config=browser_config)
        try:
            r1 = model.generate("hello")
            assert "Hello" in r1

            r2 = model.generate("what is 2+2")
            assert "4" in r2

            # Both responses should still be in the chat
            # The model should return only the LATEST response
            assert r2 != r1
        finally:
            model.close()

    def test_unknown_prompt_gets_default_response(self, browser_config: BrowserChatConfig):
        """Prompts not in the canned responses get the default."""
        model = BrowserModel(config=browser_config)
        try:
            response = model.generate("tell me about quantum physics")
            assert "don't understand" in response.lower()
        finally:
            model.close()


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 2: Typing Modes
# ══════════════════════════════════════════════════════════════════════════════

class TestTypingModes:
    """Test both fill() and type() modes for entering text.

    === PLAYWRIGHT CONCEPT: fill() vs type() deep dive ===

    fill():
        - Calls Element.value = "text" directly
        - Dispatches 'input' and 'change' events
        - INSTANT — great for tests
        - May NOT trigger keydown/keyup handlers
        - Some React/Vue components may not react to it

    type():
        - Dispatches keydown → keypress → textInput → input → keyup
        - For EACH character individually
        - SLOW but realistic
        - Works with ALL JavaScript frameworks
        - The delay= parameter adds human-like pauses
    """

    def test_fill_mode(self, browser_config: BrowserChatConfig):
        """Test with simulate_typing=False (default, uses fill())."""
        browser_config.simulate_typing = False
        model = BrowserModel(config=browser_config)
        try:
            response = model.generate("hello")
            assert "Hello" in response
        finally:
            model.close()

    def test_type_mode(self, browser_config: BrowserChatConfig):
        """Test with simulate_typing=True (uses type(), slower)."""
        browser_config.simulate_typing = True
        browser_config.typing_delay_ms = 10  # Fast for tests
        model = BrowserModel(config=browser_config)
        try:
            response = model.generate("hello")
            assert "Hello" in response
        finally:
            model.close()


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 3: Cleanup and Lifecycle
# ══════════════════════════════════════════════════════════════════════════════

class TestLifecycle:
    """Test browser lifecycle management.

    === PLAYWRIGHT CONCEPT: Resource management ===

    Browser processes are REAL OS processes. If you forget to close them:
    - They consume memory (100MB+ per browser)
    - They hold ports open
    - They can cause "too many open files" errors
    - They appear as zombie processes

    Proper cleanup is CRITICAL with Playwright.
    """

    def test_close_is_idempotent(self, browser_config: BrowserChatConfig):
        """Calling close() multiple times should not raise errors."""
        model = BrowserModel(config=browser_config)
        model.generate("hello")
        model.close()  # First close
        model.close()  # Second close — should not crash

    def test_lazy_initialization(self, browser_config: BrowserChatConfig):
        """Browser should NOT launch until generate() is called."""
        model = BrowserModel(config=browser_config)
        # Before generate(), no browser should be running
        assert model._browser is None
        assert model._page is None
        model.close()

    def test_browser_launches_on_first_generate(self, browser_config: BrowserChatConfig):
        """Browser should launch on the first generate() call."""
        model = BrowserModel(config=browser_config)
        try:
            model.generate("hello")
            assert model._browser is not None
            assert model._page is not None
        finally:
            model.close()
