"""
Playwright E2E tests for the HTML report.

=== PLAYWRIGHT CRASH COURSE ===

Playwright is a browser automation library. It launches a REAL browser
(Chromium, Firefox, or WebKit) and controls it programmatically.

Key concepts you'll learn in this file:

1. PAGE & BROWSER LIFECYCLE
   - pytest-playwright gives you a `page` fixture automatically
   - Each test gets a fresh browser context (like incognito mode)
   - The browser launches headless by default (no visible window)

2. LOCATORS (how to find elements)
   - page.locator("css-selector")      — CSS selector (like querySelector)
   - page.get_by_role("button")        — By ARIA role (accessibility-first!)
   - page.get_by_text("hello")         — By visible text content
   - page.get_by_placeholder("search") — By placeholder text
   - page.get_by_test_id("my-id")      — By data-testid attribute

   RULE: Prefer role-based and text-based locators over CSS selectors.
   They're more resilient to HTML changes and test what users actually see.

3. ASSERTIONS (expect)
   - expect(locator).to_be_visible()   — Element exists and is visible
   - expect(locator).to_have_text()    — Text content matches
   - expect(locator).to_have_count(n)  — Number of matching elements
   - expect(locator).to_have_attribute() — HTML attribute check
   - expect(page).to_have_title()      — Page title check

   Assertions AUTO-WAIT up to 5 seconds by default. If the element
   isn't ready yet, Playwright retries until it is or times out.
   This is why Playwright tests almost never need sleep() calls.

4. ACTIONS (interacting with the page)
   - locator.click()                   — Click an element
   - locator.fill("text")             — Type into an input
   - locator.press("Enter")           — Press a keyboard key
   - page.keyboard.press("Tab")       — Global keyboard events

5. AUTO-WAITING
   Playwright automatically waits for elements to be:
   - Attached to the DOM
   - Visible
   - Stable (not animating)
   - Enabled (not disabled)
   - Ready to receive events
   You almost NEVER need time.sleep(). If you're tempted to add
   a sleep, you're probably doing something wrong.

6. FIXTURES (pytest-playwright integration)
   - `page`     — A fresh Page object for each test
   - `browser`  — The shared browser instance
   - `context`  — The browser context (like an incognito session)

   We'll also write CUSTOM fixtures to generate test reports.
"""

import re
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect
from llm_eval_kit.reporters.html_report import save_html_report
from llm_eval_kit.schemas import EvalReport, EvalResult, TestCase


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES — shared setup code that runs before tests
# ══════════════════════════════════════════════════════════════════════════════

# --- What's a fixture? ---
# A fixture is a function that provides test data or setup.
# When a test function has a parameter name matching a fixture,
# pytest automatically calls the fixture and passes its return value.
#
# Example:
#   @pytest.fixture
#   def greeting():
#       return "hello"
#
#   def test_something(greeting):   # <- pytest sees "greeting" matches a fixture
#       assert greeting == "hello"  #    and automatically passes "hello" here


@pytest.fixture
def sample_report() -> EvalReport:
    """Create a realistic EvalReport with mixed pass/fail results.

    This is our test data — 5 results with different scores, tags,
    and evaluator types. Tests will verify the HTML report renders
    all of this correctly.
    """
    return EvalReport(
        model_name="test-model/v1",
        results=[
            EvalResult(
                test_case=TestCase(
                    prompt="What is the capital of France?",
                    expected="Paris",
                    tags=["geography", "factual"],
                ),
                response="Paris",
                score=1.0,
                evaluator_name="exact_match",
                reasoning="Match: 'paris' == 'paris'",
            ),
            EvalResult(
                test_case=TestCase(
                    prompt="What is 2 + 2?",
                    expected="4",
                    tags=["math", "easy"],
                ),
                response="4",
                score=1.0,
                evaluator_name="exact_match",
                reasoning="Match: '4' == '4'",
            ),
            EvalResult(
                test_case=TestCase(
                    prompt="Explain quantum computing",
                    expected="Quantum computing uses qubits",
                    tags=["science", "hard"],
                ),
                response="Quantum computing is about cats in boxes",
                score=0.3,
                evaluator_name="llm_judge",
                reasoning="Partially correct but oversimplified",
            ),
            EvalResult(
                test_case=TestCase(
                    prompt="What is the largest planet?",
                    expected="Jupiter",
                    tags=["science", "factual"],
                ),
                response="Saturn",
                score=0.0,
                evaluator_name="exact_match",
                reasoning="No match: 'saturn' != 'jupiter'",
            ),
            EvalResult(
                test_case=TestCase(
                    prompt="Who wrote Romeo and Juliet?",
                    expected="Shakespeare",
                    tags=["literature"],
                ),
                response="The play Romeo and Juliet was written by Shakespeare",
                score=1.0,
                evaluator_name="contains",
                reasoning="Found 'shakespeare' in response",
            ),
        ],
    )


@pytest.fixture
def report_path(tmp_path: Path, sample_report: EvalReport) -> Path:
    """Generate an HTML report file and return its path.

    `tmp_path` is a built-in pytest fixture that creates a temporary
    directory unique to each test invocation. Files are cleaned up
    automatically after the test run.

    This fixture DEPENDS on `sample_report` — pytest resolves the
    dependency chain automatically:
        report_path needs sample_report → creates report → creates file
    """
    html_path = tmp_path / "test_report.html"
    save_html_report(sample_report, html_path)
    return html_path


@pytest.fixture
def report_page(page: Page, report_path: Path) -> Page:
    """Open the HTML report in the Playwright browser and return the page.

    `page` is provided by pytest-playwright — it's a fresh browser tab.

    page.goto() navigates to a URL. For local files, we use file:// protocol.
    We wait for "domcontentloaded" because our report is self-contained
    (no external resources to wait for).
    """
    # Navigate to the local HTML file
    # ── CONCEPT: Navigation ──
    # page.goto() is like typing a URL in the address bar.
    # It waits for the page to reach the specified load state:
    #   - "domcontentloaded" — HTML parsed, DOM ready (fast)
    #   - "load" — all resources loaded (images, CSS, etc.)
    #   - "networkidle" — no network requests for 500ms (slowest, most thorough)
    page.goto(f"file://{report_path}", wait_until="domcontentloaded")
    return page


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 1: Basic Page Structure
# ══════════════════════════════════════════════════════════════════════════════

class TestPageStructure:
    """Verify the HTML report loads correctly and has the right structure.

    These tests cover:
    - Page title
    - Header elements
    - Dashboard cards
    - Basic layout
    """

    def test_page_title(self, report_page: Page):
        """
        ── CONCEPT: Page-level assertions ──

        expect(page).to_have_title() checks the <title> tag.
        The `re=` parameter lets you use regex for partial matching.
        This is useful when the title includes dynamic content (model name).
        """
        expect(report_page).to_have_title(re.compile("Eval Report"))

    def test_header_visible(self, report_page: Page):
        """
        ── CONCEPT: get_by_role() ──

        get_by_role() finds elements by their ARIA role. This is the
        PREFERRED way to find elements in Playwright because:
        1. It's what screen readers see (accessibility!)
        2. It's resilient to CSS class changes
        3. It focuses on what the user perceives, not implementation

        Common roles: "heading", "button", "link", "textbox",
                      "table", "row", "cell", "checkbox", etc.

        The `name=` parameter filters by accessible name (usually the text).
        """
        header = report_page.get_by_role("heading", name="llm-eval-kit")
        expect(header).to_be_visible()

    def test_dashboard_cards(self, report_page: Page):
        """
        ── CONCEPT: locator() with CSS selectors ──

        When there's no good ARIA role, CSS selectors are fine.
        .stat-card is a class on our dashboard cards.

        ── CONCEPT: to_have_count() ──
        Asserts the number of matching elements.
        We expect 4 cards: Model, Test Cases, Pass Rate, Average Score.
        """
        cards = report_page.locator(".stat-card")
        expect(cards).to_have_count(4)

    def test_dashboard_shows_model_name(self, report_page: Page):
        """
        ── CONCEPT: get_by_text() ──

        Finds elements containing the specified text.
        Simpler than CSS selectors when you just want to verify
        some text appears on the page.
        """
        expect(report_page.get_by_text("test-model/v1")).to_be_visible()

    def test_dashboard_shows_test_count(self, report_page: Page):
        """The dashboard should show our 5 test cases."""
        # We look for "5" inside the stat-card that has "Test Cases" label
        test_cases_card = report_page.locator(".stat-card", has_text="Test Cases")
        expect(test_cases_card.locator(".value")).to_have_text("5")

    def test_dashboard_shows_pass_rate(self, report_page: Page):
        """3 out of 5 results have score >= 0.5, so pass rate = 60%."""
        pass_card = report_page.locator(".stat-card", has_text="Pass Rate")
        expect(pass_card.locator(".value")).to_have_text("60%")

    def test_score_distribution_bar_visible(self, report_page: Page):
        """The score distribution bar should show passed and failed segments."""
        score_bar = report_page.locator(".score-bar")
        expect(score_bar).to_be_visible()
        # Should have both passed and failed segments
        expect(score_bar.locator(".segment.passed")).to_be_visible()
        expect(score_bar.locator(".segment.failed")).to_be_visible()


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 2: Results Table
# ══════════════════════════════════════════════════════════════════════════════

class TestResultsTable:
    """Test that the results table renders correctly with all data."""

    def test_table_has_correct_row_count(self, report_page: Page):
        """
        ── CONCEPT: Chained locators ──

        You can chain locators to narrow down selection:
        page.locator("tbody").locator("tr")
        This finds all <tr> inside <tbody>.

        Like CSS: "tbody tr" (descendant selector)
        """
        rows = report_page.locator("tbody tr")
        expect(rows).to_have_count(5)

    def test_table_shows_prompts(self, report_page: Page):
        """Each prompt from our test data should appear in the table."""
        expect(report_page.get_by_text("What is the capital of France?")).to_be_visible()
        expect(report_page.get_by_text("What is 2 + 2?")).to_be_visible()
        expect(report_page.get_by_text("Explain quantum computing")).to_be_visible()

    def test_table_shows_scores(self, report_page: Page):
        """
        ── CONCEPT: Locator filtering with has_text ──

        locator(".score-badge", has_text="1.00") finds all score badges
        showing "1.00". We use has_text for PARTIAL matching within elements.
        """
        # Should have 3 passing scores (1.00) and 2 failing
        pass_badges = report_page.locator(".score-badge.pass")
        fail_badges = report_page.locator(".score-badge.fail")
        expect(pass_badges).to_have_count(3)
        expect(fail_badges).to_have_count(2)

    def test_table_shows_tags(self, report_page: Page):
        """Tags should render as pill-shaped badges."""
        tags = report_page.locator(".tag")
        # We have tags: geography, factual, math, easy, science, hard, literature
        # Some appear multiple times (factual x2, science x2)
        # Total tag instances across all rows
        tag_count = tags.count()
        assert tag_count >= 7  # At least 7 tag pills total

    def test_results_count_text(self, report_page: Page):
        """The count text above the table should say 'Showing 5 of 5'."""
        count_text = report_page.locator("#resultsCount")
        expect(count_text).to_have_text("Showing 5 of 5 results")


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 3: Filtering
# ══════════════════════════════════════════════════════════════════════════════

class TestFiltering:
    """Test the interactive filter controls.

    This is where Playwright shines — we can CLICK buttons, TYPE in inputs,
    and DRAG sliders, then verify the table updates correctly.
    """

    def test_filter_by_passed(self, report_page: Page):
        """
        ── CONCEPT: click() ──

        locator.click() clicks an element. Playwright automatically:
        1. Scrolls the element into view
        2. Waits for it to be visible and enabled
        3. Waits for it to be stable (not animating)
        4. Clicks the center of the element

        This auto-waiting is WHY you don't need sleep() calls!
        """
        # Click the "Passed" filter button
        report_page.get_by_role("button", name="✅ Passed").click()

        # Table should now show only 3 passing results
        rows = report_page.locator("tbody tr")
        expect(rows).to_have_count(3)

        # Count text should update
        expect(report_page.locator("#resultsCount")).to_have_text(
            "Showing 3 of 5 results"
        )

    def test_filter_by_failed(self, report_page: Page):
        """Click 'Failed' filter — should show 2 rows (scores < 0.5)."""
        report_page.get_by_role("button", name="❌ Failed").click()

        rows = report_page.locator("tbody tr")
        expect(rows).to_have_count(2)

    def test_filter_by_tag(self, report_page: Page):
        """
        ── CONCEPT: Precise clicking with filter ──

        When multiple buttons have similar text, we can narrow down
        using the parent container + has_text matching.
        """
        # Click the "science" tag filter
        tag_section = report_page.locator("#tagFilters")
        tag_section.get_by_role("button", name="science").click()

        # Should show 2 results (quantum computing + largest planet)
        rows = report_page.locator("tbody tr")
        expect(rows).to_have_count(2)

    def test_filter_all_resets(self, report_page: Page):
        """Clicking 'All' after filtering should reset to show everything."""
        # First filter to only passed
        report_page.get_by_role("button", name="✅ Passed").click()
        expect(report_page.locator("tbody tr")).to_have_count(3)

        # Now click "All" in the status filter row
        status_all_btn = report_page.locator(
            '[data-filter="status"][data-value="all"]'
        )
        status_all_btn.click()

        # Should be back to 5
        expect(report_page.locator("tbody tr")).to_have_count(5)

    def test_search_filters_by_prompt(self, report_page: Page):
        """
        ── CONCEPT: fill() ──

        locator.fill("text") types text into an input field.
        Unlike type() which simulates keystrokes one at a time,
        fill() replaces the entire value at once. Faster for tests.

        ── CONCEPT: get_by_placeholder() ──
        Finds <input> elements by their placeholder text.
        More readable than locator('[placeholder="..."]').
        """
        search_box = report_page.get_by_placeholder("Search prompts & responses...")
        search_box.fill("quantum")

        # Only the quantum computing row should remain
        rows = report_page.locator("tbody tr")
        expect(rows).to_have_count(1)
        expect(report_page.get_by_text("Explain quantum computing")).to_be_visible()

    def test_search_is_case_insensitive(self, report_page: Page):
        """Search should work regardless of case."""
        report_page.get_by_placeholder("Search prompts & responses...").fill("FRANCE")
        expect(report_page.locator("tbody tr")).to_have_count(1)

    def test_search_matches_responses_too(self, report_page: Page):
        """Search should match against response text, not just prompts."""
        report_page.get_by_placeholder("Search prompts & responses...").fill("Shakespeare")
        expect(report_page.locator("tbody tr")).to_have_count(1)

    def test_score_slider(self, report_page: Page):
        """
        ── CONCEPT: Input manipulation ──

        For range sliders, we can use fill() to set the value directly,
        then dispatch an 'input' event so the JavaScript handler fires.

        This is more reliable than trying to drag the slider handle,
        which depends on exact pixel positions.
        """
        slider = report_page.locator("#scoreSlider")

        # Set slider to 50 (= min score 0.50)
        # fill() works on range inputs too!
        slider.fill("50")
        # Dispatch the input event so our JS handler fires
        slider.dispatch_event("input")

        # Slider value display should update
        expect(report_page.locator("#sliderValue")).to_have_text("0.50")

        # Only results with score >= 0.5 should remain (3 results)
        rows = report_page.locator("tbody tr")
        expect(rows).to_have_count(3)

    def test_combined_filters(self, report_page: Page):
        """
        ── CONCEPT: Testing state interactions ──

        Real users combine multiple filters. This tests that filters
        work together correctly (intersection, not union).
        """
        # Filter by "science" tag
        report_page.locator("#tagFilters").get_by_role("button", name="science").click()
        # Now also filter by "passed" status
        report_page.get_by_role("button", name="✅ Passed").click()

        # "science" tag has 2 results, but only 0 are passing (0.3 and 0.0)
        # Wait — the quantum computing result has score 0.3 (fail) and
        # largest planet has score 0.0 (fail). So 0 results!
        rows = report_page.locator("tbody tr")
        expect(rows).to_have_count(0)

        # The "no results" message should appear
        expect(report_page.locator("#noResults")).to_be_visible()

    def test_no_results_message(self, report_page: Page):
        """When filters eliminate all results, show a helpful message."""
        report_page.get_by_placeholder("Search prompts & responses...").fill(
            "xyznonexistent"
        )
        expect(report_page.locator("#noResults")).to_be_visible()
        expect(report_page.locator("#noResults")).to_have_text(
            "No results match your filters."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 4: Sorting
# ══════════════════════════════════════════════════════════════════════════════

class TestSorting:
    """Test column sorting by clicking table headers.

    Clicking a header once sorts ascending, clicking again sorts descending.
    """

    def test_sort_by_score_ascending(self, report_page: Page):
        """
        ── CONCEPT: nth() for positional selection ──

        locator.nth(0) gets the first matching element,
        locator.nth(1) gets the second, etc.
        Useful when you need elements in a specific order (like sorted rows).

        ── CONCEPT: inner_text() ──
        Returns the visible text content of an element.
        Unlike text_content(), it respects CSS (display:none, etc.).
        """
        # Click the "Score" column header to sort ascending
        report_page.locator("th[data-sort='score']").click()

        # First row should have the lowest score (0.0)
        first_score = report_page.locator("tbody tr").nth(0).locator(".score-badge")
        expect(first_score).to_have_text("0.00")

    def test_sort_by_score_descending(self, report_page: Page):
        """Click score header twice to sort descending."""
        score_header = report_page.locator("th[data-sort='score']")

        # First click → ascending
        score_header.click()
        # Second click → descending
        score_header.click()

        # First row should now have the highest score (1.0)
        first_score = report_page.locator("tbody tr").nth(0).locator(".score-badge")
        expect(first_score).to_have_text("1.00")

    def test_sort_arrow_indicator(self, report_page: Page):
        """
        ── CONCEPT: to_have_class() ──

        Assert that an element has a specific CSS class.
        Here we check that the sorted column gets the 'sorted' class.
        """
        score_header = report_page.locator("th[data-sort='score']")
        score_header.click()

        # The header should now have the 'sorted' class
        expect(score_header).to_have_class(re.compile("sorted"))


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 5: Interactive Elements
# ══════════════════════════════════════════════════════════════════════════════

class TestInteractiveElements:
    """Test expand/collapse reasoning, theme toggle, and other interactions."""

    def test_reasoning_expand_collapse(self, report_page: Page):
        """
        ── CONCEPT: to_be_hidden() / to_be_visible() ──

        These assertions check the VISIBILITY of elements.
        to_be_hidden() passes if the element is:
        - display: none
        - visibility: hidden
        - Not in the DOM at all
        - Has zero size

        Here we test the expand/collapse toggle for reasoning details.
        """
        # Initially, reasoning content should be hidden
        first_reasoning = report_page.locator(".reasoning-content").first
        expect(first_reasoning).to_be_hidden()

        # Click the "Show reasoning" toggle
        report_page.locator(".reasoning-toggle").first.click()

        # Now the reasoning should be visible
        expect(first_reasoning).to_be_visible()
        expect(first_reasoning).to_contain_text("paris")

        # Click again to collapse
        report_page.locator(".reasoning-toggle").first.click()
        expect(first_reasoning).to_be_hidden()

    def test_theme_toggle(self, report_page: Page):
        """
        ── CONCEPT: to_have_attribute() ──

        Check HTML attributes on elements.
        Our theme system uses data-theme="light"|"dark" on <html>.

        ── CONCEPT: evaluate() ──
        Runs JavaScript in the browser and returns the result.
        Useful for checking things that don't have a DOM representation.
        """
        html_el = report_page.locator("html")

        # Should start in light mode
        expect(html_el).to_have_attribute("data-theme", "light")

        # Click the theme toggle button
        report_page.locator("#themeToggle").click()

        # Should now be dark mode
        expect(html_el).to_have_attribute("data-theme", "dark")

        # Toggle back to light
        report_page.locator("#themeToggle").click()
        expect(html_el).to_have_attribute("data-theme", "light")

    def test_theme_toggle_button_icon_changes(self, report_page: Page):
        """The toggle button icon should switch between 🌙 and ☀️."""
        toggle = report_page.locator("#themeToggle")

        # Starts with moon (light mode → button offers dark mode)
        expect(toggle).to_have_text("🌙")

        toggle.click()
        # Now shows sun (dark mode → button offers light mode)
        expect(toggle).to_have_text("☀️")

    def test_filter_button_active_state(self, report_page: Page):
        """
        ── CONCEPT: to_have_class() ──

        When a filter button is clicked, it should get the 'active' class
        and the previously active button should lose it.
        """
        all_btn = report_page.locator('[data-filter="status"][data-value="all"]')
        pass_btn = report_page.get_by_role("button", name="✅ Passed")

        # "All" should start as active
        expect(all_btn).to_have_class(re.compile("active"))

        # Click "Passed"
        pass_btn.click()

        # "Passed" should now be active, "All" should not
        expect(pass_btn).to_have_class(re.compile("active"))
        # "All" should no longer have active class
        expect(all_btn).not_to_have_class(re.compile("active"))


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 6: Screenshots & Visual Testing
# ══════════════════════════════════════════════════════════════════════════════

class TestScreenshots:
    """Demonstrates Playwright's screenshot capabilities.

    ── CONCEPT: Visual Testing ──

    Screenshots are useful for:
    1. Debugging test failures (see what the page looked like)
    2. Visual regression testing (compare screenshots across runs)
    3. Documentation (auto-generate docs with real screenshots)

    Playwright can screenshot:
    - The full page (including scrolled content)
    - Just the visible viewport
    - A specific element
    """

    def test_full_page_screenshot(self, report_page: Page, tmp_path: Path):
        """
        ── CONCEPT: page.screenshot() ──

        Captures the rendered page as a PNG image.
        full_page=True captures the ENTIRE page, not just the viewport.
        This scrolls the page and stitches the screenshots together.
        """
        screenshot_path = tmp_path / "full_page.png"
        report_page.screenshot(path=str(screenshot_path), full_page=True)
        assert screenshot_path.exists()
        assert screenshot_path.stat().st_size > 0  # Not empty

    def test_element_screenshot(self, report_page: Page, tmp_path: Path):
        """
        ── CONCEPT: Element screenshots ──

        You can screenshot a specific element instead of the whole page.
        Useful for visual regression testing of individual components.
        """
        screenshot_path = tmp_path / "dashboard.png"
        report_page.locator(".dashboard").screenshot(path=str(screenshot_path))
        assert screenshot_path.exists()

    def test_dark_mode_screenshot(self, report_page: Page, tmp_path: Path):
        """Take a screenshot in dark mode — useful for visual comparison."""
        report_page.locator("#themeToggle").click()
        screenshot_path = tmp_path / "dark_mode.png"
        report_page.screenshot(path=str(screenshot_path), full_page=True)
        assert screenshot_path.exists()


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 7: Edge Cases
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def empty_report_page(self, page: Page, tmp_path: Path) -> Page:
        """Generate a report with zero results and open it."""
        report = EvalReport(model_name="empty-model", results=[])
        html_path = tmp_path / "empty_report.html"
        save_html_report(report, html_path)
        page.goto(f"file://{html_path}", wait_until="domcontentloaded")
        return page

    @pytest.fixture
    def xss_report_page(self, page: Page, tmp_path: Path) -> Page:
        """Generate a report with XSS-like content to test escaping."""
        report = EvalReport(
            model_name='<script>alert("xss")</script>',
            results=[
                EvalResult(
                    test_case=TestCase(
                        prompt='<img src=x onerror=alert("xss")>',
                        expected="safe",
                    ),
                    response='<script>document.write("hacked")</script>',
                    score=0.5,
                    evaluator_name="test",
                    reasoning="<b>bold</b>",
                ),
            ],
        )
        html_path = tmp_path / "xss_report.html"
        save_html_report(report, html_path)
        page.goto(f"file://{html_path}", wait_until="domcontentloaded")
        return page

    def test_empty_report_shows_zero_stats(self, empty_report_page: Page):
        """An empty report should show 0 for everything, not crash."""
        expect(empty_report_page.locator(".stat-card", has_text="Test Cases").locator(".value")).to_have_text("0")
        expect(empty_report_page.locator(".stat-card", has_text="Pass Rate").locator(".value")).to_have_text("0%")

    def test_empty_report_no_table_rows(self, empty_report_page: Page):
        """Empty report should show 0 rows."""
        rows = empty_report_page.locator("tbody tr")
        expect(rows).to_have_count(0)

    def test_xss_content_is_escaped(self, xss_report_page: Page):
        """
        ── CONCEPT: Security testing with Playwright ──

        We can use Playwright to verify that XSS attacks are properly
        escaped. If the <script> tags executed, they would modify the page.
        We verify they appear as TEXT, not as executed code.

        page.evaluate() runs JavaScript in the browser context.
        """
        # The script tag content should appear as text, not execute
        # If XSS worked, our test content "hacked" would appear or alerts would fire
        page_content = xss_report_page.content()
        assert "<script>alert" not in page_content or "&lt;script&gt;" in page_content

        # Verify the XSS content appears as escaped text, not executable HTML.
        # ── CONCEPT: .first / .nth() for multiple matches ──
        # get_by_text() may match multiple elements. Using .first picks
        # the first one, avoiding "strict mode violation" errors.
        expect(xss_report_page.get_by_text('alert("xss")').first).to_be_visible()


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 8: Viewport & Responsiveness
# ══════════════════════════════════════════════════════════════════════════════

class TestResponsiveness:
    """Test report layout at different viewport sizes.

    ── CONCEPT: Viewport manipulation ──

    Playwright can resize the browser viewport to test responsive designs.
    This is more reliable than resizing a real browser window because
    you control the exact pixel dimensions.
    """

    def test_mobile_viewport(self, report_page: Page):
        """
        ── CONCEPT: page.set_viewport_size() ──

        Changes the browser viewport dimensions.
        Useful for testing responsive layouts.
        """
        report_page.set_viewport_size({"width": 375, "height": 812})

        # Dashboard and table should still be visible
        expect(report_page.locator(".dashboard")).to_be_visible()
        expect(report_page.locator("table")).to_be_visible()

    def test_wide_viewport(self, report_page: Page):
        """Test on an ultra-wide viewport."""
        report_page.set_viewport_size({"width": 2560, "height": 1440})
        expect(report_page.locator(".dashboard")).to_be_visible()
        expect(report_page.locator("table")).to_be_visible()
