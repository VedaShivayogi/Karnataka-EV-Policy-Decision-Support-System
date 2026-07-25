"""
Module 9 - News Summarizer (LLM)
Collects recent EV news (via NewsAPI free tier, or a local RSS fallback)
and asks the LLM to summarize it into three buckets:
  1. New EV policies
  2. Battery technology updates
  3. Government announcements

Run:
    python modules/news_summarizer.py
"""

import sys
import os
import json
import requests
import feedparser  # pip install feedparser (no key needed, works everywhere)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.llm_client import chat

# Free fallback RSS feeds (no API key required) - used if NEWS_API_KEY is empty
RSS_FEEDS = [
    "https://news.google.com/rss/search?q=electric+vehicle+policy+India+when:14d&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=EV+battery+technology+when:14d&hl=en-IN&gl=IN&ceid=IN:en",
]


def fetch_news_newsapi():
    """Fetch via NewsAPI.org free developer tier."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": config.NEWS_QUERY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": config.NEWS_PAGE_SIZE,
        "apiKey": config.NEWS_API_KEY,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    articles = r.json().get("articles", [])
    return [
        {"title": a["title"], "source": a["source"]["name"], "desc": a.get("description") or ""}
        for a in articles
    ]


def fetch_news_rss():
    """Fallback: free Google News RSS, no key required."""
    items = []
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:10]:
            items.append(
                {
                    "title": entry.get("title", ""),
                    "source": entry.get("source", {}).get("title", "Google News"),
                    "desc": entry.get("summary", ""),
                }
            )
    return items


def fetch_news():
    if config.NEWS_API_KEY:
        try:
            return fetch_news_newsapi()
        except Exception as e:
            print(f"[warn] NewsAPI failed ({e}), falling back to RSS.")
    return fetch_news_rss()


def summarize_news(articles):
    if not articles:
        return "No articles were fetched. Check your internet connection or API key."

    corpus = "\n\n".join(
        f"- {a['title']} ({a['source']}): {a['desc']}" for a in articles
    )

    system = (
        "You are an EV policy analyst. Summarize the news into exactly three "
        "sections with headers: '## New EV Policies', '## Battery Technology Updates', "
        "'## Government Announcements'. Use crisp bullet points, cite the source name "
        "in parentheses, and skip a section if no relevant articles exist. Do not invent facts."
    )
    prompt = f"Here are recent EV-related news snippets:\n\n{corpus}\n\nSummarize as instructed."
    return chat(prompt, system=system)


def run():
    print("Fetching recent EV news...")
    articles = fetch_news()
    print(f"Fetched {len(articles)} articles. Summarizing with LLM...")
    summary = summarize_news(articles)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(config.OUTPUT_DIR, "news_summary.md")
    
    # 🔥 ಬದಲಾವಣೆ ಇಲ್ಲಿ ಮಾಡಲಾಗಿದೆ - encoding="utf-8" ಸೇರಿಸಲಾಗಿದೆ
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# EV News Summary\n\n{summary}\n")

    print(f"\nSaved summary to {out_path}\n")
    print(summary)
    return summary


if __name__ == "__main__":
    run()