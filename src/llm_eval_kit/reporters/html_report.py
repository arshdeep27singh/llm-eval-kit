"""
HTML Reporter — generates an interactive, self-contained HTML report.

The report includes:
    - Summary dashboard (model name, scores, pass rate)
    - Score distribution bar chart (pure CSS, no JS libraries)
    - Filterable, sortable results table
    - Tag-based filtering
    - Pass/Fail filter buttons
    - Score threshold slider
    - Expandable reasoning details
    - Dark/light mode toggle

The entire report is a SINGLE HTML file with embedded CSS and JS.
No external dependencies — you can open it in any browser, email it,
or host it as a static page.

Why self-contained? Because:
    1. Easy to share (just one file)
    2. Works offline
    3. Perfect target for Playwright E2E testing!

Usage:
    llm-eval-kit run examples/sample_eval.yaml --html report.html
"""

import html
import json
from pathlib import Path

from llm_eval_kit.schemas import EvalReport


def save_html_report(report: EvalReport, output_path: str | Path) -> None:
    """Generate and save an interactive HTML report.

    We build the HTML as a big template string with the eval data
    injected as a JSON blob. The JavaScript in the page reads this
    JSON and renders the interactive table/filters.

    This approach (data as JSON + client-side rendering) means:
        - Python just dumps data, doesn't build complex HTML trees
        - All interactivity (filtering, sorting) happens in the browser
        - Easy to test with Playwright!
    """
    # Convert report to JSON-serializable dict (same structure as json_report)
    report_data = {
        "model": report.model_name,
        "summary": {
            "total": report.total,
            "passed": report.passed,
            "average_score": round(report.average_score, 4),
        },
        "results": [
            {
                "index": i,
                "prompt": result.test_case.prompt,
                "expected": result.test_case.expected,
                "response": result.response,
                "score": result.score,
                "evaluator": result.evaluator_name,
                "reasoning": result.reasoning,
                "tags": result.test_case.tags,
            }
            for i, result in enumerate(report.results, start=1)
        ],
    }

    # Collect all unique tags across all test cases (for the tag filter)
    all_tags = sorted(
        {tag for result in report.results for tag in result.test_case.tags}
    )

    # Build the HTML page
    html_content = _build_html(report_data, all_tags)

    output_path = Path(output_path)
    output_path.write_text(html_content, encoding="utf-8")


def _build_html(data: dict, all_tags: list[str]) -> str:
    """Build the complete HTML string.

    The structure:
        <html>
            <head>  — CSS styles
            <body>
                <script> — Embed the eval data as JSON
                <header> — Title + summary dashboard
                <main>
                    <filters> — Tag buttons, pass/fail toggle, score slider
                    <table>   — Results rendered by JavaScript
                <footer>
                <script> — Interactive logic (filtering, sorting, etc.)
    """
    # We escape the JSON to prevent XSS — if a prompt contained </script>,
    # it would break out of the script tag. html.escape prevents that.
    escaped_json = html.escape(json.dumps(data, ensure_ascii=False), quote=False)
    escaped_tags = html.escape(json.dumps(all_tags, ensure_ascii=False), quote=False)

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eval Report — {html.escape(data["model"])}</title>
    <style>
        /* ── CSS Custom Properties (Theme Variables) ─────────────────────
           These let us do dark/light mode by just swapping variable values.
           Everything references these vars instead of hardcoded colors. */
        :root[data-theme="light"] {{
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --bg-tertiary: #e9ecef;
            --text-primary: #212529;
            --text-secondary: #6c757d;
            --border: #dee2e6;
            --accent: #4361ee;
            --accent-light: #e8ecfd;
            --success: #2ecc71;
            --danger: #e74c3c;
            --warning: #f39c12;
            --shadow: rgba(0,0,0,0.08);
        }}
        :root[data-theme="dark"] {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-tertiary: #0f3460;
            --text-primary: #e0e0e0;
            --text-secondary: #a0a0a0;
            --border: #2a2a4a;
            --accent: #4cc9f0;
            --accent-light: #1a2a3e;
            --success: #2ecc71;
            --danger: #e74c3c;
            --warning: #f39c12;
            --shadow: rgba(0,0,0,0.3);
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-secondary);
            color: var(--text-primary);
            line-height: 1.6;
            transition: background 0.3s, color 0.3s;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}

        /* ── Header ──────────────────────────────────────────────────── */
        header {{
            background: var(--bg-primary);
            border-bottom: 1px solid var(--border);
            padding: 1.5rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 4px var(--shadow);
        }}
        header h1 {{
            font-size: 1.5rem;
            font-weight: 600;
        }}
        header h1 span {{
            color: var(--accent);
        }}
        .header-controls {{
            display: flex;
            gap: 1rem;
            align-items: center;
        }}

        /* ── Theme Toggle Button ─────────────────────────────────────── */
        .theme-toggle {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.5rem 1rem;
            cursor: pointer;
            font-size: 1rem;
            color: var(--text-primary);
            transition: all 0.2s;
        }}
        .theme-toggle:hover {{
            background: var(--accent);
            color: white;
        }}

        /* ── Summary Dashboard ───────────────────────────────────────── */
        .dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }}
        .stat-card {{
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 2px 8px var(--shadow);
            transition: transform 0.2s;
        }}
        .stat-card:hover {{
            transform: translateY(-2px);
        }}
        .stat-card .label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}
        .stat-card .value {{
            font-size: 2rem;
            font-weight: 700;
        }}
        .stat-card .value.score-good {{ color: var(--success); }}
        .stat-card .value.score-ok {{ color: var(--warning); }}
        .stat-card .value.score-bad {{ color: var(--danger); }}

        /* ── Score Distribution Bar ──────────────────────────────────── */
        .score-bar-container {{
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px var(--shadow);
        }}
        .score-bar-container h3 {{
            margin-bottom: 1rem;
            font-size: 1rem;
            color: var(--text-secondary);
        }}
        .score-bar {{
            display: flex;
            height: 32px;
            border-radius: 8px;
            overflow: hidden;
            background: var(--bg-tertiary);
        }}
        .score-bar .segment {{
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 600;
            color: white;
            transition: width 0.5s ease;
        }}
        .score-bar .segment.passed {{ background: var(--success); }}
        .score-bar .segment.failed {{ background: var(--danger); }}

        /* ── Filters Section ─────────────────────────────────────────── */
        .filters {{
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px var(--shadow);
        }}
        .filters h3 {{
            font-size: 1rem;
            color: var(--text-secondary);
            margin-bottom: 1rem;
        }}
        .filter-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            align-items: center;
            margin-bottom: 1rem;
        }}
        .filter-row:last-child {{
            margin-bottom: 0;
        }}
        .filter-label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            min-width: 80px;
            font-weight: 500;
        }}

        /* Filter buttons (tags, pass/fail) */
        .filter-btn {{
            padding: 0.35rem 0.85rem;
            border: 1px solid var(--border);
            border-radius: 20px;
            background: var(--bg-secondary);
            color: var(--text-primary);
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.2s;
        }}
        .filter-btn:hover {{
            border-color: var(--accent);
            color: var(--accent);
        }}
        .filter-btn.active {{
            background: var(--accent);
            color: white;
            border-color: var(--accent);
        }}

        /* Score slider */
        .score-slider {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        .score-slider input[type="range"] {{
            flex: 1;
            max-width: 200px;
            accent-color: var(--accent);
        }}
        .score-slider .slider-value {{
            font-weight: 600;
            min-width: 2.5rem;
            text-align: center;
            color: var(--accent);
        }}

        /* Search box */
        .search-box {{
            padding: 0.5rem 1rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--bg-secondary);
            color: var(--text-primary);
            font-size: 0.9rem;
            width: 250px;
            transition: border-color 0.2s;
        }}
        .search-box:focus {{
            outline: none;
            border-color: var(--accent);
        }}

        /* ── Results Table ───────────────────────────────────────────── */
        .results-table-wrapper {{
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px var(--shadow);
        }}
        .results-count {{
            padding: 1rem 1.5rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        thead th {{
            background: var(--bg-secondary);
            padding: 0.75rem 1rem;
            text-align: left;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            border-bottom: 2px solid var(--border);
            cursor: pointer;
            user-select: none;
            white-space: nowrap;
        }}
        thead th:hover {{
            color: var(--accent);
        }}
        thead th .sort-arrow {{
            margin-left: 0.25rem;
            opacity: 0.3;
        }}
        thead th.sorted .sort-arrow {{
            opacity: 1;
            color: var(--accent);
        }}
        tbody tr {{
            border-bottom: 1px solid var(--border);
            transition: background 0.15s;
        }}
        tbody tr:hover {{
            background: var(--accent-light);
        }}
        td {{
            padding: 0.75rem 1rem;
            font-size: 0.9rem;
            vertical-align: top;
        }}
        td.prompt-cell {{
            max-width: 300px;
        }}
        td.response-cell {{
            max-width: 250px;
            word-break: break-word;
        }}

        /* Score badge */
        .score-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-weight: 700;
            font-size: 0.85rem;
        }}
        .score-badge.pass {{ background: #d4edda; color: #155724; }}
        .score-badge.fail {{ background: #f8d7da; color: #721c24; }}
        :root[data-theme="dark"] .score-badge.pass {{ background: #1a3a2a; color: #2ecc71; }}
        :root[data-theme="dark"] .score-badge.fail {{ background: #3a1a1a; color: #e74c3c; }}

        /* Tag pills */
        .tag {{
            display: inline-block;
            padding: 0.15rem 0.5rem;
            background: var(--accent-light);
            color: var(--accent);
            border-radius: 10px;
            font-size: 0.75rem;
            margin: 0.1rem;
        }}

        /* Expandable reasoning */
        .reasoning-toggle {{
            cursor: pointer;
            color: var(--accent);
            font-size: 0.8rem;
            text-decoration: underline;
        }}
        .reasoning-content {{
            display: none;
            margin-top: 0.5rem;
            padding: 0.5rem;
            background: var(--bg-secondary);
            border-radius: 6px;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}
        .reasoning-content.expanded {{
            display: block;
        }}

        /* ── Footer ──────────────────────────────────────────────────── */
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }}

        /* ── No results message ──────────────────────────────────────── */
        .no-results {{
            padding: 3rem;
            text-align: center;
            color: var(--text-secondary);
        }}
    </style>
</head>
<body>

    <!-- Header with title and theme toggle -->
    <header>
        <h1>📊 <span>llm-eval-kit</span> Report</h1>
        <div class="header-controls">
            <button class="theme-toggle" id="themeToggle" aria-label="Toggle dark mode">🌙</button>
        </div>
    </header>

    <div class="container">

        <!-- Summary Dashboard — 4 stat cards -->
        <div class="dashboard" id="dashboard">
            <!-- Filled by JavaScript -->
        </div>

        <!-- Score Distribution Bar -->
        <div class="score-bar-container">
            <h3>Score Distribution</h3>
            <div class="score-bar" id="scoreBar">
                <!-- Filled by JavaScript -->
            </div>
        </div>

        <!-- Filters -->
        <div class="filters">
            <h3>Filters</h3>

            <!-- Search row -->
            <div class="filter-row">
                <span class="filter-label">Search:</span>
                <input type="text" class="search-box" id="searchBox"
                       placeholder="Search prompts & responses...">
            </div>

            <!-- Status filter -->
            <div class="filter-row">
                <span class="filter-label">Status:</span>
                <button class="filter-btn active" data-filter="status" data-value="all">All</button>
                <button class="filter-btn" data-filter="status" data-value="pass">✅ Passed</button>
                <button class="filter-btn" data-filter="status" data-value="fail">❌ Failed</button>
            </div>

            <!-- Tag filter -->
            <div class="filter-row" id="tagFilters">
                <span class="filter-label">Tags:</span>
                <button class="filter-btn active" data-filter="tag" data-value="all">All</button>
                <!-- Tag buttons added by JavaScript -->
            </div>

            <!-- Score threshold slider -->
            <div class="filter-row">
                <span class="filter-label">Min Score:</span>
                <div class="score-slider">
                    <input type="range" id="scoreSlider" min="0" max="100" value="0" step="5">
                    <span class="slider-value" id="sliderValue">0.00</span>
                </div>
            </div>
        </div>

        <!-- Results Table -->
        <div class="results-table-wrapper">
            <div class="results-count" id="resultsCount">Showing all results</div>
            <table>
                <thead>
                    <tr>
                        <th data-sort="index"># <span class="sort-arrow">↕</span></th>
                        <th data-sort="prompt">Prompt <span class="sort-arrow">↕</span></th>
                        <th data-sort="expected">Expected <span class="sort-arrow">↕</span></th>
                        <th data-sort="response">Response <span class="sort-arrow">↕</span></th>
                        <th data-sort="score">Score <span class="sort-arrow">↕</span></th>
                        <th>Tags</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody id="resultsBody">
                    <!-- Filled by JavaScript -->
                </tbody>
            </table>
            <div class="no-results" id="noResults" style="display:none;">
                No results match your filters.
            </div>
        </div>
    </div>

    <footer>
        Generated by <strong>llm-eval-kit</strong>
    </footer>

    <!-- ── DATA INJECTION ──────────────────────────────────────────────
         We embed the eval data as a JSON string inside a <script> tag.
         The JavaScript below reads this and renders everything.
         This is the bridge between Python (generates this file) and
         the browser (makes it interactive). -->
    <script id="evalData" type="application/json">{escaped_json}</script>
    <script id="tagData" type="application/json">{escaped_tags}</script>

    <script>
        // ══════════════════════════════════════════════════════════════════
        // INTERACTIVE REPORT LOGIC
        // ══════════════════════════════════════════════════════════════════

        // ── 1. Parse embedded data ──────────────────────────────────────
        const data = JSON.parse(
            document.getElementById('evalData').textContent
        );
        const allTags = JSON.parse(
            document.getElementById('tagData').textContent
        );

        // ── 2. State management ─────────────────────────────────────────
        // All filter state lives here. When any filter changes, we
        // re-render the table. Simple and predictable.
        let state = {{
            statusFilter: 'all',   // 'all', 'pass', or 'fail'
            tagFilter: 'all',      // 'all' or a specific tag string
            minScore: 0.0,         // 0.0 to 1.0
            searchQuery: '',       // text search
            sortField: 'index',    // which column to sort by
            sortAsc: true,         // ascending or descending
        }};

        // ── 3. Render Dashboard ─────────────────────────────────────────
        function renderDashboard() {{
            const s = data.summary;
            const avgClass = s.average_score >= 0.7 ? 'score-good'
                           : s.average_score >= 0.5 ? 'score-ok' : 'score-bad';
            const passRate = s.total > 0 ? ((s.passed / s.total) * 100).toFixed(0) : 0;

            document.getElementById('dashboard').innerHTML = `
                <div class="stat-card">
                    <div class="label">Model</div>
                    <div class="value" style="font-size:1.2rem">${{data.model}}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Test Cases</div>
                    <div class="value">${{s.total}}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Pass Rate</div>
                    <div class="value ${{avgClass}}">${{passRate}}%</div>
                </div>
                <div class="stat-card">
                    <div class="label">Average Score</div>
                    <div class="value ${{avgClass}}">${{s.average_score.toFixed(2)}}</div>
                </div>
            `;
        }}

        // ── 4. Render Score Distribution Bar ────────────────────────────
        function renderScoreBar() {{
            const s = data.summary;
            const passW = s.total > 0 ? (s.passed / s.total * 100) : 0;
            const failW = 100 - passW;

            document.getElementById('scoreBar').innerHTML = `
                ${{passW > 0 ? `<div class="segment passed" style="width:${{passW}}%">${{s.passed}} passed</div>` : ''}}
                ${{failW > 0 ? `<div class="segment failed" style="width:${{failW}}%">${{s.total - s.passed}} failed</div>` : ''}}
            `;
        }}

        // ── 5. Render Tag Filter Buttons ────────────────────────────────
        function renderTagFilters() {{
            const container = document.getElementById('tagFilters');
            allTags.forEach(tag => {{
                const btn = document.createElement('button');
                btn.className = 'filter-btn';
                btn.dataset.filter = 'tag';
                btn.dataset.value = tag;
                btn.textContent = tag;
                container.appendChild(btn);
            }});
        }}

        // ── 6. Filter & Sort Logic ──────────────────────────────────────
        function getFilteredResults() {{
            let results = [...data.results];

            // Apply status filter
            if (state.statusFilter === 'pass') {{
                results = results.filter(r => r.score >= 0.5);
            }} else if (state.statusFilter === 'fail') {{
                results = results.filter(r => r.score < 0.5);
            }}

            // Apply tag filter
            if (state.tagFilter !== 'all') {{
                results = results.filter(r => r.tags.includes(state.tagFilter));
            }}

            // Apply score threshold
            if (state.minScore > 0) {{
                results = results.filter(r => r.score >= state.minScore);
            }}

            // Apply search query
            if (state.searchQuery) {{
                const q = state.searchQuery.toLowerCase();
                results = results.filter(r =>
                    r.prompt.toLowerCase().includes(q) ||
                    r.response.toLowerCase().includes(q) ||
                    (r.expected || '').toLowerCase().includes(q)
                );
            }}

            // Apply sorting
            results.sort((a, b) => {{
                let valA = a[state.sortField] ?? '';
                let valB = b[state.sortField] ?? '';
                if (typeof valA === 'string') valA = valA.toLowerCase();
                if (typeof valB === 'string') valB = valB.toLowerCase();
                if (valA < valB) return state.sortAsc ? -1 : 1;
                if (valA > valB) return state.sortAsc ? 1 : -1;
                return 0;
            }});

            return results;
        }}

        // ── 7. Render Table Rows ────────────────────────────────────────
        function renderTable() {{
            const filtered = getFilteredResults();
            const tbody = document.getElementById('resultsBody');
            const noResults = document.getElementById('noResults');
            const countEl = document.getElementById('resultsCount');

            countEl.textContent = `Showing ${{filtered.length}} of ${{data.results.length}} results`;

            if (filtered.length === 0) {{
                tbody.innerHTML = '';
                noResults.style.display = 'block';
                return;
            }}

            noResults.style.display = 'none';
            tbody.innerHTML = filtered.map(r => {{
                const scoreClass = r.score >= 0.5 ? 'pass' : 'fail';
                const tags = r.tags.map(t => `<span class="tag">${{t}}</span>`).join(' ');
                const reasoningId = `reasoning-${{r.index}}`;

                return `<tr data-index="${{r.index}}">
                    <td>${{r.index}}</td>
                    <td class="prompt-cell">${{escapeHtml(r.prompt)}}</td>
                    <td>${{escapeHtml(r.expected || '-')}}</td>
                    <td class="response-cell">${{escapeHtml(r.response)}}</td>
                    <td><span class="score-badge ${{scoreClass}}">${{r.score.toFixed(2)}}</span></td>
                    <td>${{tags || '-'}}</td>
                    <td>
                        <span class="reasoning-toggle" data-target="${{reasoningId}}">
                            Show reasoning
                        </span>
                        <div class="reasoning-content" id="${{reasoningId}}">
                            <strong>Evaluator:</strong> ${{r.evaluator}}<br>
                            <strong>Reasoning:</strong> ${{escapeHtml(r.reasoning || 'No reasoning provided')}}
                        </div>
                    </td>
                </tr>`;
            }}).join('');
        }}

        // ── 8. Escape HTML (prevent XSS in displayed data) ─────────────
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}

        // ── 9. Event Listeners ──────────────────────────────────────────

        // Theme toggle
        document.getElementById('themeToggle').addEventListener('click', () => {{
            const html = document.documentElement;
            const current = html.getAttribute('data-theme');
            const next = current === 'light' ? 'dark' : 'light';
            html.setAttribute('data-theme', next);
            document.getElementById('themeToggle').textContent = next === 'light' ? '🌙' : '☀️';
        }});

        // Filter buttons (status + tag)
        document.addEventListener('click', (e) => {{
            if (!e.target.classList.contains('filter-btn')) return;

            const filter = e.target.dataset.filter;
            const value = e.target.dataset.value;

            if (filter === 'status') {{
                state.statusFilter = value;
                // Update active state for status buttons only
                document.querySelectorAll('[data-filter="status"]').forEach(
                    btn => btn.classList.toggle('active', btn.dataset.value === value)
                );
            }} else if (filter === 'tag') {{
                state.tagFilter = value;
                document.querySelectorAll('[data-filter="tag"]').forEach(
                    btn => btn.classList.toggle('active', btn.dataset.value === value)
                );
            }}

            renderTable();
        }});

        // Score slider
        document.getElementById('scoreSlider').addEventListener('input', (e) => {{
            state.minScore = parseInt(e.target.value) / 100;
            document.getElementById('sliderValue').textContent = state.minScore.toFixed(2);
            renderTable();
        }});

        // Search box
        document.getElementById('searchBox').addEventListener('input', (e) => {{
            state.searchQuery = e.target.value;
            renderTable();
        }});

        // Sortable columns
        document.querySelectorAll('thead th[data-sort]').forEach(th => {{
            th.addEventListener('click', () => {{
                const field = th.dataset.sort;
                if (state.sortField === field) {{
                    state.sortAsc = !state.sortAsc;
                }} else {{
                    state.sortField = field;
                    state.sortAsc = true;
                }}
                // Update sort arrow styles
                document.querySelectorAll('thead th').forEach(h => h.classList.remove('sorted'));
                th.classList.add('sorted');
                th.querySelector('.sort-arrow').textContent = state.sortAsc ? '↑' : '↓';
                renderTable();
            }});
        }});

        // Reasoning expand/collapse (event delegation)
        document.addEventListener('click', (e) => {{
            if (!e.target.classList.contains('reasoning-toggle')) return;
            const target = document.getElementById(e.target.dataset.target);
            if (target) {{
                target.classList.toggle('expanded');
                e.target.textContent = target.classList.contains('expanded')
                    ? 'Hide reasoning' : 'Show reasoning';
            }}
        }});

        // ── 10. Initial render ──────────────────────────────────────────
        renderDashboard();
        renderScoreBar();
        renderTagFilters();
        renderTable();
    </script>
</body>
</html>"""
