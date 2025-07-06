from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for, send_file
import feedparser
import json
import os
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from langdetect import detect
import google.generativeai as genai
import pyttsx3

load_dotenv()

app = Flask(__name__, static_url_path='/static')

# Fix: Register Python's enumerate in Jinja
app.jinja_env.globals.update(enumerate=enumerate)

# --- Constants for filenames ---
DATA_FILE = 'feeds.json'
FAV_FILE = 'favorites.json'
SETTINGS_FILE = 'settings.json'
READ_STATUS_FILE = 'read.json'

# --- Configure Gemini API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Utility Functions ---

def load_json(filename):
    """Loads data from a JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    if filename == SETTINGS_FILE:
        return {'theme': 'light', 'refresh': 1, 'translate_enabled': True, 'categories': []}
    return []

def load_feeds():
    """Alias for loading feeds from DATA_FILE."""
    return load_json(DATA_FILE)

def load_settings():
    """Alias for loading settings from SETTINGS_FILE."""
    return load_json(SETTINGS_FILE)

def load_read_status():
    data = load_json(READ_STATUS_FILE)
    return set(data) if isinstance(data, list) else set()

def save_read_status(status):
    save_json(READ_STATUS_FILE, list(status))

def save_json(filename, data):
    """Saves data to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def save_feeds(feeds):
    """Alias for saving feeds to DATA_FILE."""
    save_json(DATA_FILE, feeds)

def translate_text(text, target_lang='English'):
    """Translates text using the Gemini API."""
    try:
        if not text.strip():
            return text
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        prompt = f"Translate the following into {target_lang}:\n\n{text}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Translation Error: {e}")
        return text

def clean_html(raw_html):
    """Removes HTML tags from a string."""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=' ', strip=True)

def get_ai_summary(url):
    """Summarizes an article from a URL using Gemini."""
    if not url or not GEMINI_API_KEY:
        return "No link provided or Gemini API key is not configured."
    prompt = (
        f"Please provide a two-part summary of the article at this URL: {url}\n\n"
        f"1. **Short Summary:** A concise overview in 2-3 sentences.\n"
        f"2. **Detailed Summary:** A more thorough breakdown of the key points."
    )
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while summarizing: {str(e)}"

def fetch_feed_entries(url):
    """Fetches entries from an RSS feed URL."""
    parsed = feedparser.parse(url)
    return parsed.entries[:5]  # Limit to 5 entries per feed, as in original

def parse_entry(entry, source, category):
    """Parses a feed entry into an article dictionary."""
    settings = load_settings()
    title = entry.title
    summary = clean_html(entry.get('summary', 'No summary available.'))
    link = entry.link
    published = entry.get('published', 'No date available')

    if settings.get('translate_enabled'):
        try:
            if title.strip() and detect(title) != 'en':
                title = translate_text(title)
            if summary.strip() and detect(summary) != 'en':
                summary = translate_text(summary)
        except Exception as e:
            print(f"Language detection/translation failed: {e}")

    image = ''
    if 'media_content' in entry and entry.media_content:
        image = entry.media_content[0].get('url', '')
    elif 'links' in entry:
        for link_obj in entry.links:
            if link_obj.get('type', '').startswith('image/'):
                image = link_obj.get('href', '')
                break

    return {
        'title': title,
        'summary': summary,
        'link': link,
        'published': published,
        'source': source,
        'image': image,
        'category': category
    }

def load_articles():
    """Loads all articles from all feeds."""
    feeds = load_feeds()
    articles = []
    for feed in feeds:
        entries = fetch_feed_entries(feed['url'])
        for entry in entries:
            articles.append(parse_entry(entry, feed['url'], feed.get('category', 'Uncategorized')))
    return articles

def merge_duplicates(articles):
    """Merges duplicate articles by title, counting occurrences."""
    title_counts = {}
    for article in articles:
        title_counts[article['title']] = title_counts.get(article['title'], 0) + 1

    unique_articles = []
    seen_titles = set()
    for article in articles:
        if article['title'] not in seen_titles:
            article['count'] = title_counts[article['title']]
            unique_articles.append(article)
            seen_titles.add(article['title'])

    return unique_articles

# --- Main Routes ---

@app.route('/')
def index():
    """Main page route, displays all articles from feeds with filtering by feed or category."""
    selected_feed = request.args.get("feed")
    selected_category = request.args.get('category')
    all_feeds = load_json(DATA_FILE)
    all_articles = load_articles()
    settings = load_settings()
    categories = sorted(set(feed.get("category", "Uncategorized") for feed in all_feeds))

    # Filter articles by selected feed
    if selected_feed is not None and selected_feed.isdigit():
        selected_feed = int(selected_feed)
        if 0 <= selected_feed < len(all_feeds):
            selected_feed_url = all_feeds[selected_feed]["url"]
            articles = [a for a in all_articles if a["source"] == selected_feed_url]
        else:
            articles = all_articles
            selected_feed = None
    # Filter by category if no feed is selected
    elif selected_category:
        articles = [a for a in all_articles if a["category"] == selected_category]
    else:
        articles = all_articles
        selected_feed = None

    # Add unread status to each article
    read_links = load_read_status()
    for article in articles:
        article["unread"] = article["link"] not in read_links

    # Handle search query
    query = request.args.get('q', '').lower()
    if query:
        articles = [a for a in articles if query in a['title'].lower() or query in a['summary'].lower()]

    # Sort articles by publication date
    articles = sorted(articles, key=lambda x: x.get('published', ''), reverse=True)

    return render_template(
        "index.html",
        articles=articles,
        feeds=all_feeds,
        settings=settings,
        categories=categories,
        selected_feed=selected_feed,
        selected_category=selected_category
    )

@app.route('/toggle_read', methods=['POST'])
def toggle_read():
    """Toggles the read status of an article."""
    link = request.form['link']
    read_articles = load_read_status()
    if link in read_articles:
        read_articles.remove(link)
    else:
        read_articles.add(link)
    save_read_status(read_articles)
    return '', 204

@app.route('/add_category', methods=['POST'])
def add_category():
    """Adds a new category to settings."""
    settings = load_settings()
    categories = settings.get('categories', [])
    new_cat = request.form['category'].strip()
    if new_cat and new_cat not in categories:
        categories.append(new_cat)
        settings['categories'] = categories
        save_json(SETTINGS_FILE, settings)
    return redirect(url_for('settings'))

@app.route('/add_feed', methods=['POST'])
def add_feed():
    """Adds a new RSS feed."""
    name = request.form['name']
    url = request.form['url']
    category = request.form.get('category', 'Uncategorized')
    feeds = load_feeds()
    if not any(f['url'] == url for f in feeds):
        feeds.append({'name': name, 'url': url, 'category': category})
        save_feeds(feeds)
    return redirect(url_for('index'))

@app.route('/delete_feed/<int:index>')
def delete_feed(index):
    """Deletes an RSS feed by its index."""
    feeds = load_feeds()
    if 0 <= index < len(feeds):
        feeds.pop(index)
        save_feeds(feeds)
    return redirect(url_for('index'))

@app.route('/rename_feed/<int:idx>', methods=['POST'])
def rename_feed(idx):
    new_name = request.form.get('new_name')
    feeds = load_feeds()
    if 0 <= idx < len(feeds):
        feeds[idx]['name'] = new_name
        save_feeds(feeds)
    return redirect('/settings')

@app.route('/rename_feed', methods=['POST'])
def rename_feed_form():
    index = int(request.form['index'])
    new_name = request.form['new_name']
    feeds = load_feeds()
    if 0 <= index < len(feeds):
        feeds[index]['name'] = new_name
        save_feeds(feeds)
    return redirect('/settings')

@app.route('/delete_all_feeds', methods=['POST'])
def delete_all_feeds():
    save_feeds([])  # Clear all feeds
    return redirect('/settings')

# --- Favorite Routes ---

@app.route('/favorite', methods=['POST'])
def favorite():
    """Adds an article to the favorites list."""
    favs = load_json(FAV_FILE)
    article = {
        'title': request.form['title'],
        'link': request.form['link'],
        'source': request.form['source'],
        'published': request.form['published'],
        'summary': request.form['summary'],
        'image': request.form.get('image', ''),
        'category': request.form.get('category', 'Uncategorized')
    }
    if not any(f['link'] == article['link'] for f in favs):
        favs.append(article)
        save_json(FAV_FILE, favs)
    return redirect(url_for('index'))

@app.route('/favorites')
def favorites():
    """Displays the list of favorited articles."""
    favs = load_json(FAV_FILE)
    settings = load_settings()
    categories = settings.get('categories', [])
    return render_template('favorites.html', favorites=favs, settings=settings, categories=categories)

@app.route('/remove_favorite/<int:index>')
def remove_favorite(index):
    """Removes an article from the favorites list."""
    favs = load_json(FAV_FILE)
    if 0 <= index < len(favs):
        favs.pop(index)
        save_json(FAV_FILE, favs)
    return redirect(url_for('favorites'))

# --- Settings and Import/Export ---

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Handles application settings."""
    current_settings = load_settings()
    if request.method == 'POST':
        current_settings['theme'] = request.form.get('theme', 'light')
        current_settings['refresh'] = int(request.form.get('refresh', 1))
        current_settings['translate_enabled'] = 'translate_enabled' in request.form
        save_json(SETTINGS_FILE, current_settings)
        return redirect(url_for('settings'))
    return render_template('settings.html', settings=current_settings)

@app.route('/export_opml')
def export_opml():
    """Exports the current feed list to an OPML file."""
    feeds = load_feeds()
    root = ET.Element('opml', version='1.0')
    body = ET.SubElement(root, 'body')
    for feed in feeds:
        outline = ET.SubElement(body, 'outline', type='rss', text=feed['name'], xmlUrl=feed['url'])
        if 'category' in feed:
            outline.set('category', feed['category'])
    tree = ET.ElementTree(root)
    opml_filename = 'feeds.opml'
    tree.write(opml_filename, encoding='utf-8', xml_declaration=True)
    return send_file(opml_filename, as_attachment=True)

@app.route('/import_opml', methods=['POST'])
def import_opml():
    """Imports feeds from an uploaded OPML file."""
    if 'opml_file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['opml_file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    tree = ET.parse(file)
    root = tree.getroot()
    feeds = load_feeds()
    existing_urls = {f['url'] for f in feeds}
    
    for outline in root.findall('.//outline[@xmlUrl]'):
        url = outline.attrib.get('xmlUrl')
        name = outline.attrib.get('text', url)
        category = outline.attrib.get('category', 'Uncategorized')
        if url and url not in existing_urls:
            feeds.append({'name': name, 'url': url, 'category': category})
            existing_urls.add(url)
            
    save_feeds(SETTINGS_FILE, feeds)
    return redirect(url_for('index'))

# --- Article Action Routes ---

@app.route('/translate', methods=['POST'])
def translate():
    """Translates a given piece of text."""
    summary = request.form['summary']
    target_lang = request.form.get('lang', 'Urdu')
    settings = load_settings()
    translation = translate_text(summary, target_lang)
    return render_template('translate.html', original=summary, translated=translation, lang=target_lang, settings=settings)

@app.route('/narrate', methods=['POST'])
def narrate():
    """Narrates a summary using text-to-speech."""
    summary = request.form['summary']
    if not os.path.exists('static'):
        os.makedirs('static')
    
    try:
        engine = pyttsx3.init()
        engine.save_to_file(summary, 'static/speech.mp3')
        engine.runAndWait()
        return send_file('static/speech.mp3', as_attachment=False, mimetype='audio/mpeg')
    except Exception as e:
        print(f"Narration Error: {e}")
        return "Error generating audio.", 500

@app.route('/view_article')
def view_article():
    """Displays a single article in a focused view."""
    settings = load_settings()
    return render_template('view_article.html', settings=settings, **request.args)

@app.route('/summarize', methods=['POST'])
def summarize():
    url = request.form.get('url')
    summary = get_ai_summary(url)
    return render_template('summary.html', summary=summary, title=request.form.get('title'), source=request.form.get('source'), published=request.form.get('published'))

@app.route('/voice', methods=['POST'])
def voice():
    """Searches articles based on a voice query."""
    query = request.form.get(query, '').lower()
    feeds = load_feeds()
    settings = load_settings()
    categories = settings.get('categories', [])
    articles = []

    if not query:
        return redirect(url_for('index'))

    for feed in feeds:
        entries = fetch_feed_entries(feed['url'])
        for entry in entries:
            article = parse_entry(entry, feed['name'], feed.get('category', 'Uncategorized'))
            if query in article['title'].lower() or query in article['summary'].lower():
                articles.append(article)

    return render_template('index.html', articles=articles, feeds=feeds, settings=settings, categories=categories, query=query)

# --- Main execution ---
if __name__ == '__main__':
    app.run(debug=True)
