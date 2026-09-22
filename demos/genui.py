"""Generative UI: from one plain-language request the model decides a dashboard's domain, layout, lead metric, chart, panels, theme
and accent in ONE call (ten decisions, each with its probability); a shadcn-style renderer (Tailwind + Chart.js) draws the page.

    python demos/genui.py --endpoint http://localhost:3009 --out demos/out/genui --screenshots
    python demos/genui.py --from-decisions demos/out/genui/decisions.json --out demos/out/genui --screenshots   # re-render only

Nothing in the page is scripted per request: change the sentence and the decisions change the page.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path

from common import Player

REQUESTS = {
    "support_dashboard": "Dashboard for my support team lead: ticket backlog by priority, who is overloaded, and response times over the last two weeks.",
    "saas_growth": "A growth dashboard for our SaaS: signups, activation, monthly recurring revenue trend and the top acquisition channels.",
    "expense_approvals": "Internal tool where a finance manager reviews employee expense reports one by one and approves or rejects them.",
    "pricing": "A page where visitors compare our three pricing plans (Starter, Team, Enterprise) and pick one.",
    "ops_status": "A public status page for our API: current incidents, uptime for the last 90 days, and a way to subscribe to updates.",
    "sales_pipeline": "A sales manager's view of this quarter's pipeline: deals by stage, forecast against quota, and the reps who are behind.",
}

DOMAIN = {
    "support": "customer support and tickets",
    "product": "product growth, signups and revenue",
    "finance": "expenses, budgets and approvals",
    "sales": "pipeline, deals and quota",
    "operations": "uptime, incidents and infrastructure",
    "marketing": "campaigns, traffic and conversion",
}
LAYOUT = {
    "sidebar_dashboard": "left navigation sidebar, panels in a grid",
    "topnav_dashboard": "top navigation bar, panels in a grid",
    "table_page": "one big data table with a toolbar",
    "review_page": "one record at a time with approve / reject controls",
    "landing": "headline, plan cards and a call to action",
    "status_page": "a status banner, uptime bars and an incident list",
}
LEAD = {
    "open_tickets": None,
    "median_response": None,
    "signups": None,
    "mrr": None,
    "pending_approvals": None,
    "pipeline_value": None,
    "uptime": None,
    "conversion_rate": None,
}
CHART = {
    "line_trend": "a line chart over time",
    "area_trend": "an area chart over time",
    "bar_categories": "bars comparing categories",
    "donut_share": "a donut of shares",
    "uptime_bars": "90 daily uptime bars",
    "none": "no chart",
}
PANEL = {
    "data_table": "a table of records",
    "people_load": "a list of people with load bars",
    "activity_feed": "recent activity",
    "incident_list": "current incidents",
    "funnel": "a funnel of stages",
    "plan_cards": "pricing plan cards",
    "review_form": "one record with approve and reject buttons",
}
TONE = {"playful": "friendly copy", "neutral": "plain copy", "formal": "businesslike copy"}
ACCENT = {"zinc": None, "blue": None, "green": None, "violet": None, "orange": None, "rose": None}

QUESTIONS = {
    "domain": {"type": "choice", "instructions": "Which domain is this page for?", "criteria": DOMAIN},
    "layout": {"type": "choice", "instructions": "Which page layout fits this request best?", "criteria": LAYOUT},
    "lead": {"type": "choice", "instructions": "Which metric should be the first, biggest number on the page?", "criteria": LEAD},
    "chart": {"type": "choice", "instructions": "Which chart should the page have?", "criteria": CHART},
    "panel": {"type": "choice", "instructions": "Which panel should sit next to the chart?", "criteria": PANEL},
    "cta": {"type": "noul", "instructions": "Should the page have one prominent call-to-action button?"},
    "dark": {"type": "noul", "instructions": "Should the page use a dark theme?"},
    "density": {"type": "score", "instructions": "How dense should the layout be?", "criteria": ["spacious", "balanced", "dense"]},
    "tone": {"type": "choice", "instructions": "What tone should the copy have?", "criteria": TONE},
    "accent": {"type": "choice", "instructions": "Which accent colour suits this page?", "criteria": ACCENT},
}

# ---- sample data per domain (labelled as sample data on the page)
KPIS = {
    "support": [
        ("open_tickets", "Open tickets", "148", "+12 today"),
        ("median_response", "Median first reply", "2h 40m", "-18% vs last week"),
        ("overdue", "Overdue", "23", "SLA breached"),
        ("csat", "CSAT", "94%", "+1.2 pts"),
    ],
    "product": [
        ("signups", "Signups", "3,482", "+8% this month"),
        ("activation", "Activation", "61%", "+3 pts"),
        ("mrr", "MRR", "$184k", "+$11k"),
        ("conversion_rate", "Trial to paid", "12.4%", "flat"),
    ],
    "finance": [
        ("pending_approvals", "Pending approvals", "37", "9 over $1,000"),
        ("spend", "Spend this month", "$212k", "84% of budget"),
        ("avg_report", "Average report", "$486", ""),
        ("flagged", "Flagged", "4", "policy exceptions"),
    ],
    "sales": [
        ("pipeline_value", "Pipeline", "$2.4M", "quarter to date"),
        ("forecast", "Forecast", "$1.1M", "of $1.4M quota"),
        ("won", "Won", "$640k", "18 deals"),
        ("at_risk", "At risk", "7", "no activity 14 days"),
    ],
    "operations": [
        ("uptime", "Uptime 90 days", "99.98%", "2 incidents"),
        ("latency", "p95 latency", "212 ms", "-9 ms"),
        ("incidents", "Open incidents", "1", "degraded: webhooks"),
        ("subscribers", "Subscribers", "1,204", ""),
    ],
    "marketing": [
        ("conversion_rate", "Conversion", "3.8%", "+0.4 pts"),
        ("traffic", "Sessions", "412k", "+11%"),
        ("cac", "CAC", "$84", "-6%"),
        ("campaigns", "Live campaigns", "9", ""),
    ],
}
SERIES = {
    "support": ("Tickets per day", [42, 51, 47, 63, 58, 49, 38, 44, 57, 61, 55, 48, 52, 46]),
    "product": ("Signups per day", [88, 92, 110, 105, 124, 131, 118, 140, 152, 149, 163, 171, 168, 180]),
    "finance": ("Spend per week ($k)", [38, 44, 52, 41, 47, 55, 49, 58, 61, 57, 63, 60, 66, 71]),
    "sales": ("Pipeline by stage ($k)", [820, 640, 410, 290, 180, 90]),
    "operations": ("Uptime %", [100] * 60 + [99.2] + [100] * 20 + [99.7] + [100] * 8),
    "marketing": ("Sessions per day (k)", [26, 28, 31, 29, 33, 35, 34, 38, 41, 39, 43, 45, 44, 47]),
}
LABELS = {"sales": ["Lead", "Qualified", "Proposal", "Negotiation", "Verbal", "Closing"]}
TABLE = {
    "support": (
        ["Ticket", "Subject", "Priority", "Owner", "Age"],
        [
            ["#4415", "Refund not received", "high", "Mia", "3d"],
            ["#4412", "Login loop on iOS", "urgent", "Sam", "5h"],
            ["#4409", "Invoice address", "low", "Ana", "1d"],
            ["#4404", "Export fails", "high", "Sam", "2d"],
            ["#4401", "Wrong plan shown", "medium", "Lee", "6h"],
        ],
    ),
    "product": (
        ["Channel", "Signups", "Activation", "Paid"],
        [
            ["Organic search", "1,240", "64%", "148"],
            ["Referral", "610", "71%", "96"],
            ["Paid social", "880", "49%", "61"],
            ["Newsletter", "412", "77%", "72"],
            ["Partners", "340", "58%", "40"],
        ],
    ),
    "finance": (
        ["Report", "Employee", "Amount", "Policy", "Status"],
        [
            ["#2291", "J. Ortiz", "$612.40", "ok", "pending"],
            ["#2290", "A. Khan", "$1,480.00", "needs manager", "pending"],
            ["#2288", "M. Chen", "$96.10", "ok", "approved"],
            ["#2287", "S. Berg", "$2,210.00", "receipt missing", "flagged"],
            ["#2285", "L. Rossi", "$310.00", "ok", "approved"],
        ],
    ),
    "sales": (
        ["Deal", "Rep", "Stage", "Value", "Close"],
        [
            ["Northwind", "Priya", "Negotiation", "$240k", "Oct 3"],
            ["Contoso", "Dan", "Proposal", "$180k", "Oct 14"],
            ["Fabrikam", "Lee", "Qualified", "$95k", "Nov 2"],
            ["Tailspin", "Priya", "Verbal", "$310k", "Sep 30"],
            ["Litware", "Ana", "Lead", "$60k", "Nov 20"],
        ],
    ),
    "operations": (
        ["Service", "Status", "Uptime 90d", "p95"],
        [
            ["API", "operational", "99.99%", "180 ms"],
            ["Webhooks", "degraded", "99.84%", "1.2 s"],
            ["Dashboard", "operational", "100%", "310 ms"],
            ["Auth", "operational", "99.98%", "95 ms"],
        ],
    ),
    "marketing": (
        ["Campaign", "Channel", "Spend", "Conversions", "CAC"],
        [
            ["Fall launch", "Paid social", "$18k", "214", "$84"],
            ["Webinar series", "Email", "$2k", "96", "$21"],
            ["Brand search", "Search", "$9k", "180", "$50"],
            ["Partner promo", "Referral", "$4k", "71", "$56"],
        ],
    ),
}
PEOPLE = [("Sam", 92), ("Mia", 74), ("Ana", 58), ("Lee", 41), ("Priya", 33)]
FEED = ["Sam closed #4412", "Mia reassigned #4415 to Lee", "New ticket #4418 from Acme", "Ana replied to #4409", "SLA breached on #4404"]
INCIDENTS = [("Webhook delivery delays", "degraded", "investigating, 14 min ago"), ("Elevated API errors in eu-west", "resolved", "yesterday, 38 min")]
PLANS = [
    ("Starter", "$0", ["1 project", "Community support", "Basic analytics"]),
    ("Team", "$29 / seat", ["Unlimited projects", "Email support", "Shared dashboards"]),
    ("Enterprise", "Talk to us", ["SSO and audit log", "Dedicated support", "Custom retention"]),
]
NAV = {
    "support": ["Overview", "Queues", "People", "SLAs", "Reports"],
    "product": ["Overview", "Acquisition", "Activation", "Revenue", "Cohorts"],
    "finance": ["Overview", "Approvals", "Budgets", "Policies", "Exports"],
    "sales": ["Overview", "Pipeline", "Forecast", "Reps", "Accounts"],
    "operations": ["Status", "Incidents", "Uptime", "Subscribe"],
    "marketing": ["Overview", "Campaigns", "Channels", "Attribution", "Audiences"],
}
ACCENTS = {
    "zinc": ("#18181b", "#fafafa"),
    "blue": ("#2563eb", "#ffffff"),
    "green": ("#16a34a", "#ffffff"),
    "violet": ("#7c3aed", "#ffffff"),
    "orange": ("#ea580c", "#ffffff"),
    "rose": ("#e11d48", "#ffffff"),
}
TITLE = {"playful": "Here's how things look", "neutral": "Overview", "formal": "Executive summary"}


def kpi_cards(domain: str, lead: str, n: int) -> str:
    rows = KPIS[domain]
    rows = sorted(rows, key=lambda r: r[0] != lead)[:n]  # the lead metric first
    out = []
    for i, (_, label, value, note) in enumerate(rows):
        big = "text-4xl" if i == 0 else "text-2xl"
        out.append(
            f'<div class="card p-5"><div class="text-sm text-muted">{label}</div><div class="{big} font-semibold tracking-tight mt-1">{value}</div><div class="text-xs text-muted mt-1">{note}</div></div>'
        )
    return f'<div class="grid gap-4" style="grid-template-columns:repeat({n},minmax(0,1fr))">{"".join(out)}</div>'


def chart_card(domain: str, kind: str, accent: str) -> str:
    if kind == "none":
        return ""
    title, data = SERIES[domain] if kind != "uptime_bars" else SERIES["operations"]
    if kind == "uptime_bars":
        bars = "".join(f'<i class="uptime" style="background:{"#16a34a" if v == 100 else "#f59e0b" if v > 99.5 else "#e11d48"}"></i>' for v in data)
        return f'<div class="card p-5"><div class="font-medium mb-3">{title}, last 90 days</div><div class="flex gap-[2px] h-10 items-end">{bars}</div><div class="flex justify-between text-xs text-muted mt-2"><span>90 days ago</span><span>today</span></div></div>'
    labels = LABELS.get(domain) if kind == "bar_categories" and domain in LABELS else [f"D{i + 1}" for i in range(len(data))]
    if kind == "donut_share":
        labels, data = (["Organic", "Referral", "Paid", "Email", "Partners"], [38, 19, 27, 10, 6])
    cfg = {
        "type": {"line_trend": "line", "area_trend": "line", "bar_categories": "bar", "donut_share": "doughnut"}[kind],
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": title,
                    "data": data,
                    "borderColor": accent,
                    "backgroundColor": accent + ("33" if kind == "area_trend" else "cc" if kind != "donut_share" else ""),
                    "fill": kind == "area_trend",
                    "tension": 0.35,
                    "borderWidth": 2,
                    "pointRadius": 0,
                }
            ],
        },
        "options": {
            "plugins": {"legend": {"display": kind == "donut_share"}},
            "scales": {} if kind == "donut_share" else {"x": {"grid": {"display": False}}, "y": {"grid": {"color": "rgba(127,127,127,.15)"}}},
            "maintainAspectRatio": False,
        },
    }
    if kind == "donut_share":
        cfg["data"]["datasets"][0]["backgroundColor"] = [accent, accent + "b0", accent + "80", accent + "55", accent + "30"]
    return f'<div class="card p-5"><div class="font-medium mb-3">{title}</div><div style="height:240px"><canvas id="chart"></canvas></div><script>new Chart(document.getElementById("chart"),{json.dumps(cfg)})</script></div>'


def panel_card(domain: str, kind: str, cta: bool) -> str:
    if kind == "data_table":
        head, rows = TABLE[domain]
        ths = "".join(f"<th class='pb-2 pr-4 font-medium'>{h}</th>" for h in head)
        body = "".join("<tr>" + "".join(f'<td class="py-2 pr-4">{c}</td>' for c in r) + "</tr>" for r in rows)
        return (
            '<div class="card p-5 overflow-x-auto"><div class="font-medium mb-3">Records</div><table class="w-full text-sm">'
            f'<thead><tr class="text-muted text-left">{ths}</tr></thead><tbody class="divide-y divide-border">{body}</tbody></table></div>'
        )
    if kind == "people_load":
        items = "".join(
            f'<div class="flex items-center gap-3 py-2"><div class="w-16 text-sm">{n}</div><div class="flex-1 h-2 rounded bg-muted"><div class="h-2 rounded bg-accent" style="width:{p}%"></div></div><div class="w-10 text-right text-sm text-muted">{p}%</div></div>'
            for n, p in PEOPLE
        )
        return f'<div class="card p-5"><div class="font-medium mb-2">Load by person</div>{items}</div>'
    if kind == "activity_feed":
        return (
            '<div class="card p-5"><div class="font-medium mb-2">Recent activity</div>'
            + "".join(f'<div class="py-2 text-sm border-b border-border last:border-0">{x}</div>' for x in FEED)
            + "</div>"
        )
    if kind == "incident_list":
        return (
            '<div class="card p-5"><div class="font-medium mb-2">Incidents</div>'
            + "".join(
                f'<div class="py-3 border-b border-border last:border-0"><div class="flex items-center gap-2"><span class="badge {s}">{s}</span><span class="font-medium text-sm">{t}</span></div><div class="text-xs text-muted mt-1">{w}</div></div>'
                for t, s, w in INCIDENTS
            )
            + "</div>"
        )
    if kind == "funnel":
        stages = [("Visitors", 100), ("Signups", 42), ("Activated", 26), ("Paid", 12)]
        return (
            '<div class="card p-5"><div class="font-medium mb-2">Funnel</div>'
            + "".join(
                f'<div class="py-1"><div class="flex justify-between text-sm"><span>{n}</span><span class="text-muted">{p}%</span></div><div class="h-2 rounded bg-muted mt-1"><div class="h-2 rounded bg-accent" style="width:{p}%"></div></div></div>'
                for n, p in stages
            )
            + "</div>"
        )
    if kind == "plan_cards":
        cards = "".join(
            f'<div class="card p-6 flex flex-col"><div class="font-medium">{n}</div><div class="text-3xl font-semibold tracking-tight my-2">{p}</div><ul class="text-sm text-muted space-y-1 mb-4">{"".join(f"<li>{f}</li>" for f in fs)}</ul><button class="btn mt-auto">Choose {n}</button></div>'
            for n, p, fs in PLANS
        )
        return f'<div class="grid gap-4 md:grid-cols-3">{cards}</div>'
    if kind == "review_form":
        return '<div class="card p-6"><div class="text-sm text-muted">Expense report #2290</div><div class="text-xl font-semibold mt-1">A. Khan, client visit, Berlin</div><table class="text-sm my-4 w-full"><tr><td class="py-1 text-muted">Taxi</td><td class="text-right">$64.00</td></tr><tr><td class="py-1 text-muted">Hotel, 2 nights</td><td class="text-right">$1,120.00</td></tr><tr><td class="py-1 text-muted">Meals</td><td class="text-right">$296.00</td></tr><tr class="border-t border-border font-medium"><td class="py-1">Total</td><td class="text-right">$1,480.00</td></tr></table><div class="text-xs text-muted mb-4">Over $1,000: needs manager approval.</div><div class="flex gap-2"><button class="btn">Approve</button><button class="btn-outline">Reject</button><button class="btn-outline">Ask for receipt</button></div></div>'
    return ""


def render(request: str, d: dict) -> str:
    def g(k: str) -> str:
        return d[k]["choice"]

    domain, layout, chart, panel, tone = g("domain"), g("layout"), g("chart"), g("panel"), g("tone")
    dark, cta = d["dark"]["noul"] >= 0.5, d["cta"]["noul"] >= 0.5
    density = ("spacious", "balanced", "dense")[min(2, max(0, round(d["density"]["score"])))]
    accent, on_accent = ACCENTS[g("accent")]
    gap = {"spacious": "gap-6 p-8", "balanced": "gap-4 p-6", "dense": "gap-3 p-4"}[density]
    kpis = kpi_cards(domain, g("lead"), 3 if density == "spacious" else 4)
    chart_html = chart_card(domain, chart, accent)
    panel_html = panel_card(domain, panel, cta)
    nav = "".join(f'<a class="nav {"active" if i == 0 else ""}">{n}</a>' for i, n in enumerate(NAV[domain]))
    title = html.escape(request.split(":")[0].split(".")[0])
    button = (
        f'<button class="btn">{ {"support": "New ticket", "product": "Invite team", "finance": "Review next", "sales": "Add deal", "operations": "Subscribe to updates", "marketing": "New campaign"}[domain] }</button>'
        if cta
        else ""
    )
    header = f'<div class="flex items-center justify-between mb-6"><div><div class="text-sm text-muted">{TITLE[tone]}</div><h1 class="text-2xl font-semibold tracking-tight">{title}</h1></div>{button}</div>'
    if layout == "landing":
        main = f'<section class="text-center py-10"><h1 class="text-4xl font-semibold tracking-tight">{title}</h1><p class="text-muted mt-3 text-lg">Pick the plan that fits. Change any time.</p>{"<div class=mt-6>" + button + "</div>" if cta else ""}</section>{panel_card(domain, "plan_cards", cta)}<div class="mt-6">{chart_html}</div>'
        shell = f'<main class="mx-auto max-w-5xl {gap}">{main}</main>'
    elif layout == "status_page":
        banner = '<div class="card p-4 mb-4 flex items-center gap-3"><span class="badge operational">operational</span><span class="font-medium">All systems operational, one degraded component</span></div>'
        shell = f'<main class="mx-auto max-w-4xl {gap}">{header}{banner}{chart_card(domain, "uptime_bars", accent)}<div class="mt-4">{panel_card(domain, "incident_list", cta)}</div><div class="mt-4">{kpis}</div></main>'
    elif layout == "table_page":
        toolbar = '<div class="flex gap-2 mb-4"><input class="input" placeholder="Search"><span class="chip">priority: all</span><span class="chip">owner: all</span><span class="chip">last 14 days</span></div>'
        shell = f'<main class="mx-auto max-w-6xl {gap}">{header}{toolbar}{panel_card(domain, "data_table", cta)}<div class="mt-4 grid gap-4 md:grid-cols-2">{chart_html}{kpis}</div></main>'
    elif layout == "review_page":
        shell = f'<main class="mx-auto max-w-3xl {gap}">{header}{panel_card(domain, "review_form", cta)}<div class="mt-4">{kpis}</div></main>'
    else:
        grid = (
            f'<div class="grid gap-4 md:grid-cols-3"><div class="md:col-span-2">{chart_html}</div><div>{panel_html}</div></div>'
            if chart != "none"
            else panel_html
        )
        body = f"{header}{kpis}<div class='mt-4'>{grid}</div>"
        if layout == "sidebar_dashboard":
            shell = f'<div class="flex min-h-screen"><aside class="w-56 border-r border-border p-4 hidden md:block"><div class="font-semibold mb-4">Acme</div>{nav}</aside><main class="flex-1 {gap}">{body}</main></div>'
        else:
            shell = f'<header class="border-b border-border px-6 py-3 flex items-center gap-6"><div class="font-semibold">Acme</div><nav class="flex gap-4">{nav}</nav></header><main class="mx-auto max-w-6xl {gap}">{body}</main>'
    bg, fg, card, muted, border = ("#09090b", "#fafafa", "#18181b", "#a1a1aa", "#27272a") if dark else ("#fafafa", "#09090b", "#ffffff", "#71717a", "#e4e4e7")
    css = f"""
:root{{--accent:{accent};--on:{on_accent};--bg:{bg};--fg:{fg};--card:{card};--muted:{muted};--border:{border}}}
body{{background:var(--bg);color:var(--fg);font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;-webkit-font-smoothing:antialiased}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:12px;box-shadow:0 1px 2px rgba(0,0,0,.05)}}.text-muted{{color:var(--muted)}}.bg-muted{{background:{border}}}.bg-accent{{background:var(--accent)}}.border-border{{border-color:var(--border)}}.divide-border>:not([hidden])~:not([hidden]){{border-color:var(--border)}}
.btn{{background:var(--accent);color:var(--on);border-radius:8px;padding:8px 14px;font-size:14px;font-weight:500}}.btn-outline{{border:1px solid var(--border);border-radius:8px;padding:8px 14px;font-size:14px;font-weight:500}}
.nav{{display:block;padding:6px 10px;border-radius:6px;font-size:14px;color:var(--muted)}}.nav.active{{background:{border};color:var(--fg)}}.input{{border:1px solid var(--border);background:var(--card);border-radius:8px;padding:8px 12px;font-size:14px;width:260px}}
.chip{{border:1px solid var(--border);border-radius:999px;padding:6px 10px;font-size:13px;color:var(--muted)}}.badge{{font-size:11px;border-radius:999px;padding:2px 8px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}}.badge.operational,.badge.resolved{{background:#dcfce7;color:#166534}}.badge.degraded{{background:#fef3c7;color:#92400e}}
.uptime{{flex:1;height:100%;border-radius:2px;display:block}}
"""
    return f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet"><script src="https://cdn.tailwindcss.com"></script><script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script><style>{css}</style></head><body>{shell}<div class="text-xs text-muted text-center py-4">Sample data. Layout, metric, chart, panel, theme and accent chosen by OpenJev from the request in one call.</div></body></html>'


def screenshot(html_path: Path, png_path: Path) -> bool:
    chrome = next(
        (
            c
            for c in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", shutil.which("google-chrome"), shutil.which("chromium"))
            if c and Path(c).exists()
        ),
        None,
    )
    if not chrome:
        return False
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--window-size=1200,760",
            "--virtual-time-budget=4000",
            f"--screenshot={png_path}",
            html_path.resolve().as_uri(),
        ],
        capture_output=True,
        timeout=90,
    )
    return png_path.exists()


def summarize(answers: dict) -> dict:
    return {k: (v.get("choice") if "choice" in v else v.get("noul") if "noul" in v else round(v.get("score"), 2)) for k, v in answers.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="http://localhost:3000")
    ap.add_argument("--model", default="openjev")
    ap.add_argument("--out", default="demos/out/genui")
    ap.add_argument("--screenshots", action="store_true")
    ap.add_argument("--from-decisions", help="re-render from a saved decisions.json without calling the model")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.from_decisions:
        rows = json.loads(Path(a.from_decisions).read_text())
    else:
        player = Player(a.endpoint, a.model, name="model")
        rows = []
        for name, request in REQUESTS.items():
            answers, ms = player.ask(request, QUESTIONS)
            rows.append({"case": name, "request": request, "answers": answers, "decisions": summarize(answers), "ms": round(ms)})
            print(f"{name:18s} {round(ms):5d} ms  {json.dumps(rows[-1]['decisions'])}", flush=True)
        (out / "decisions.json").write_text(json.dumps(rows, indent=1) + "\n")
        print(json.dumps({"cases": len(rows), "avg_ms": round(player.avg_ms())}))
    for r in rows:
        (out / f"{r['case']}.html").write_text(render(r["request"], r["answers"]))
        if a.screenshots:
            screenshot(out / f"{r['case']}.html", out / f"{r['case']}.png")


if __name__ == "__main__":
    main()
