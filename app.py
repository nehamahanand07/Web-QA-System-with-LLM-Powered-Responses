import streamlit as st
import time, requests
from urllib.parse import urljoin, urldefrag
from bs4 import BeautifulSoup
from gpt4all import GPT4All
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- CRAWLER -----------------
HEADERS = {"User-Agent": "MySimpleCrawler/1.0"}
SKIP_EXT = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".mp4", ".mp3")

def normalize_url(url):
    return urldefrag(url)[0].strip()

def get_links(base_url, html):
    soup = BeautifulSoup(html, "html.parser")
    return [urljoin(base_url, a["href"]) for a in soup.find_all("a", href=True)]

def clean_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)

def crawl_site(seed_url, max_pages=3, delay=1):
    seed_url = normalize_url(seed_url)
    visited = set()
    to_visit = [seed_url]
    pages = []

    while to_visit and len(pages) < max_pages:
        url = to_visit.pop(0)
        if url in visited or url.lower().endswith(SKIP_EXT):
            continue
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue
            text = clean_text(resp.text)
            pages.append({"url": url, "text": text})
            visited.add(url)
            to_visit.extend([normalize_url(link) for link in get_links(url, resp.text) if link not in visited])
            time.sleep(delay)
        except Exception as e:
            st.warning(f"Error fetching {url}: {e}")
            continue
    return pages

# ----------------- CHUNKER -----------------
def chunk_text(text, max_words=220, overlap=40):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+max_words]))
        if i + max_words >= len(words):
            break
        i += max_words - overlap
    return chunks

def pages_to_chunks(pages, max_words=220, overlap=40):
    docs = []
    for p in pages:
        chs = chunk_text(p["text"], max_words=max_words, overlap=overlap)
        for idx, ch in enumerate(chs):
            docs.append({"url": p["url"], "content": ch, "chunk_id": f'{p["url"]}#chunk={idx}'})
    return docs

# ----------------- SIMPLE VECTOR STORE -----------------
class SimpleVectorStore:
    def __init__(self, index_dir=None):
        self.docs = []
        self.vectorizer = TfidfVectorizer()
        self.vectors = None

    def build(self, docs):
        self.docs = docs
        texts = [d["content"] for d in docs]
        self.vectors = self.vectorizer.fit_transform(texts)

    def save(self):
        pass

    def load(self):
        pass

    def search(self, query, top_k=3):
        if self.vectors is None or len(self.docs) == 0:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.vectors)[0]
        results = sorted(zip(self.docs, sims), key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:top_k]]

# ----------------- RETRIEVER -----------------
class Retriever:
    def __init__(self, index_dir=None):
        self.store = SimpleVectorStore(index_dir)
        self.store.load()
    def get_context(self, question, top_k=3):
        return self.store.search(question, top_k=top_k)

# ----------------- GPT4ALL -----------------
SYSTEM_PROMPT = "Answer the question only using the provided context. Cite sources. If not found, say 'Not found on this site.'"
model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")

def format_context(chunks):
    formatted = []
    for i, c in enumerate(chunks, 1):
        formatted.append(f"[{i}] {c['content']}\n(Source: {c['url']})")
    return "\n\n".join(formatted)

def answer_with_gpt4all(question, chunks):
    context = format_context(chunks)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
    return model.generate(prompt, max_tokens=400)

# ----------------- STREAMLIT APP -----------------
st.title("🌐 Web Crawler + GPT4All QA")

with st.form("crawler_form"):
    url_input = st.text_input("Enter the URL to crawl:")
    question_input = st.text_input("Enter your question about this site:")
    max_pages = st.number_input("Max pages to crawl:", min_value=1, max_value=10, value=1)
    submit_btn = st.form_submit_button("Crawl & Get Answer")

if submit_btn:
    if not url_input or not question_input:
        st.warning("Please enter both a URL and a question!")
    else:
        with st.spinner("Crawling website..."):
            pages = crawl_site(url_input, max_pages=max_pages)
            if not pages:
                st.error("No pages crawled. Check the URL or try again.")
            else:
                chunks = pages_to_chunks(pages, max_words=100, overlap=20)
                
                store = SimpleVectorStore()
                store.build(chunks)
                retriever = Retriever()
                retriever.store = store  # Assign the built store
                
                st.success(f"Crawled {len(pages)} pages and created {len(chunks)} chunks!")

                relevant_chunks = retriever.get_context(question_input, top_k=3)
                answer = answer_with_gpt4all(question_input, relevant_chunks)
                
                st.write("### Answer:")
                st.write(answer)

                st.write("### Top Context Chunks Used:")
                for ch in relevant_chunks:
                    st.write(f"- ({ch['url']}) {ch['content'][:200]}...")

# ✅ Debug line
st.write("✅ Streamlit reached the end of the script")
