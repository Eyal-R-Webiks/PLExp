import requests
import wikipedia
import os
import time
import json


# ─── PARAMS ────────────────────────────────────────────────────────────────────

MIN_CHARS   = 1000
MAX_CHARS   = 30000
TARGET_DOCS = 300
OUTPUT_DIR  = "wikipedia_articles"

# How many QIDs to fetch per category (we need ~15 per category to hit 300 total
# after filtering by char length and skipping fetch errors)
QID_LIMIT_PER_CATEGORY = 25

# ─── CATEGORIES ────────────────────────────────────────────────────────────────
# Each value is a SPARQL query that returns items with a Hebrew Wikipedia article.
# All queries were tested against query.wikidata.org before including here.

CATEGORY_QUERIES = {
    "scientists":        "SELECT DISTINCT ?item WHERE { ?item wdt:P31 wd:Q5; wdt:P106/wdt:P279* wd:Q901. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
    "politicians":       "SELECT DISTINCT ?item WHERE { ?item wdt:P31 wd:Q5; wdt:P106 wd:Q82955. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
    "historical_events": "SELECT DISTINCT ?item WHERE { ?item wdt:P31/wdt:P279* wd:Q13418847. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
    "wars":              "SELECT DISTINCT ?item WHERE { ?item wdt:P31/wdt:P279* wd:Q198. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
    "battles":           "SELECT DISTINCT ?item WHERE { ?item wdt:P31/wdt:P279* wd:Q178561. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
    "military_officers": "SELECT DISTINCT ?item WHERE { ?item wdt:P31 wd:Q5; wdt:P106 wd:Q189290. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
    "countries":         "SELECT DISTINCT ?item WHERE { ?item wdt:P31 wd:Q3624078. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
    "cities":            "SELECT DISTINCT ?item WHERE { ?item wdt:P31 wd:Q515. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
    "athletes":          "SELECT DISTINCT ?item WHERE { ?item wdt:P31 wd:Q5; wdt:P106 wd:Q2066131. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
    "sports_teams":      "SELECT DISTINCT ?item WHERE { ?item wdt:P31/wdt:P279* wd:Q847017. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
    "films":             "SELECT DISTINCT ?item WHERE { ?item wdt:P31 wd:Q11424. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
    "books":             "SELECT DISTINCT ?item WHERE { ?item wdt:P31/wdt:P279* wd:Q7725634. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
    "writers":           "SELECT DISTINCT ?item WHERE { ?item wdt:P31 wd:Q5; wdt:P106 wd:Q36180. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
    "musicians":         "SELECT DISTINCT ?item WHERE { ?item wdt:P31 wd:Q5; wdt:P106/wdt:P279* wd:Q639669. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
    "philosophers":      "SELECT DISTINCT ?item WHERE { ?item wdt:P31 wd:Q5; wdt:P106 wd:Q4964182. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
    "religion":          "SELECT DISTINCT ?item WHERE { ?item wdt:P31/wdt:P279* wd:Q9174. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
    "species":           "SELECT DISTINCT ?item WHERE { ?item wdt:P31 wd:Q16521. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
    "astronomy":         "SELECT DISTINCT ?item WHERE { ?item wdt:P31/wdt:P279* wd:Q6999. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
    "diseases":          "SELECT DISTINCT ?item WHERE { ?item wdt:P31/wdt:P279* wd:Q12136. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
    "companies":         "SELECT DISTINCT ?item WHERE { ?item wdt:P31/wdt:P279* wd:Q4830453. ?sl schema:about ?item; schema:isPartOf <https://he.wikipedia.org/>. } LIMIT 25",
}

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_API    = "https://www.wikidata.org/w/api.php"
REQUEST_HEADERS = {"User-Agent": "hebrew-wikipedia-collector/1.0 (research project)"}


# ─── FUNCTIONS ─────────────────────────────────────────────────────────────────

def fetch_qids_for_category(category_name, query):
    try:
        r = requests.get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers={**REQUEST_HEADERS, "Accept": "application/sparql-results+json"},
            timeout=45,
        )
        bindings = r.json()["results"]["bindings"]
        qids = [b["item"]["value"].split("/")[-1] for b in bindings]
        print(f"  {category_name}: {len(qids)} QIDs from Wikidata")
        return qids
    except Exception as e:
        print(f"  {category_name}: FAILED ({e})")
        return []


def get_hebrew_titles_batch(qids):
    # Wikidata API accepts up to 50 IDs per call
    r = requests.get(
        WIKIDATA_API,
        params={
            "action":     "wbgetentities",
            "ids":        "|".join(qids),
            "props":      "sitelinks",
            "sitefilter": "hewiki",
            "format":     "json",
        },
        headers=REQUEST_HEADERS,
        timeout=30,
    )
    entities = r.json().get("entities", {})
    result = {}
    for qid, data in entities.items():
        title = data.get("sitelinks", {}).get("hewiki", {}).get("title")
        if title:
            result[qid] = title
    return result


def fetch_article_text(hebrew_title):
    try:
        page = wikipedia.page(hebrew_title, auto_suggest=False)
        return page.content
    except wikipedia.exceptions.DisambiguationError:
        return None
    except wikipedia.exceptions.PageError:
        return None
    except Exception:
        return None


def save_article(qid, category, title, text):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Use QID as filename so it's always valid and traceable
    safe_filename = f"{qid}.txt"
    filepath = os.path.join(OUTPUT_DIR, safe_filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"qid: {qid}\n")
        f.write(f"category: {category}\n")
        f.write(f"title: {title}\n")
        f.write(f"chars: {len(text)}\n")
        f.write("---\n")
        f.write(text)


# ─── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    wikipedia.set_lang("he")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_collected = 0
    seen_qids = set()
    summary = {}

    print("=== Step 1: Fetching QIDs from Wikidata ===\n")
    category_qids = {}
    for category, query in CATEGORY_QUERIES.items():
        qids = fetch_qids_for_category(category, query)
        category_qids[category] = qids
        time.sleep(0.5)  # stay polite to Wikidata

    print(f"\n=== Step 2: Resolving QIDs to Hebrew titles ===\n")
    category_titles = {}
    for category, qids in category_qids.items():
        if not qids:
            continue
        # fetch in one batch (all are <= 25, well under the 50-item API limit)
        qid_to_title = get_hebrew_titles_batch(qids)
        category_titles[category] = qid_to_title
        print(f"  {category}: {len(qid_to_title)} have Hebrew articles")
        time.sleep(0.3)

    print(f"\n=== Step 3: Fetching articles from Hebrew Wikipedia ===\n")
    for category, qid_to_title in category_titles.items():
        collected_this_category = 0

        for qid, title in qid_to_title.items():
            if total_collected >= TARGET_DOCS:
                break
            if qid in seen_qids:
                continue

            text = fetch_article_text(title)
            if text is None:
                continue

            char_count = len(text)
            if char_count < MIN_CHARS or char_count > MAX_CHARS:
                continue

            save_article(qid, category, title, text)
            seen_qids.add(qid)
            total_collected += 1
            collected_this_category += 1
            print(f"  [{total_collected}/{TARGET_DOCS}] {category} | {title} ({char_count} chars)")

            time.sleep(0.2)  # be polite to Wikipedia API

        summary[category] = collected_this_category
        if total_collected >= TARGET_DOCS:
            break

    print(f"\n=== Done ===")
    print(f"Total articles collected: {total_collected}")
    print(f"Saved to: {OUTPUT_DIR}/")
    print("\nBreakdown by category:")
    for cat, count in summary.items():
        print(f"  {cat}: {count}")

    # Save a manifest so it's easy to see what was collected
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    manifest = []
    for fname in os.listdir(OUTPUT_DIR):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(OUTPUT_DIR, fname)
        with open(fpath, encoding="utf-8") as f:
            lines = f.read().splitlines()
        meta = {l.split(": ", 1)[0]: l.split(": ", 1)[1] for l in lines[:4] if ": " in l}
        manifest.append(meta)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nManifest saved to: {manifest_path}")
