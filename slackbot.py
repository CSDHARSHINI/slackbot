import streamlit as st
import pandas as pd
import re
import requests
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from fpdf import FPDF
import matplotlib.pyplot as plt

# Title
st.title(" Slacktbot: Keyword Intelligence Tool")

# Choose input method
st.sidebar.header("Input")
input_type = st.sidebar.radio("Select Input Type", ["Manual Entry", "Upload CSV", "Paste Keywords"])

# Keyword input section
keywords = []

# 1 Manual Entry (3–4 fields)
if input_type == "Manual Entry":
    st.subheader("Enter 3–4 Keywords")
    kw1 = st.text_input("Keyword 1", placeholder="e.g.")
    kw2 = st.text_input("Keyword 2", placeholder="e.g.")
    kw3 = st.text_input("Keyword 3", placeholder="e.g.")
    kw4 = st.text_input("Keyword 4 ", placeholder="e.g.")

    keywords = [k for k in [kw1, kw2, kw3, kw4] if k.strip()]

# 2 Upload CSV
elif input_type == "Upload CSV":
    upload = st.file_uploader("Upload a CSV containing keywords", type=["csv"])
    if upload:
        df = pd.read_csv(upload)
        keywords = df.iloc[:, 0].dropna().tolist()

# 3 Paste keywords
elif input_type == "Paste Keywords":
    raw_input = st.text_area("Paste keywords here (one per line):")
    if raw_input:
        keywords = [k.strip() for k in raw_input.split('\n') if k.strip()]

# Proceed only if there are keywords
if keywords:
    st.success(f"{len(keywords)} keywords received!")
    st.write("Input Keywords:", keywords)

    # Clean and normalize
    def clean_kw(kw):
        return re.sub(r'\W+', ' ', kw.lower()).strip()

    cleaned = list(set([clean_kw(k) for k in keywords]))
    st.write("Cleaned Keywords:", cleaned)

    # Sentence Embedding Model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(cleaned)

    # Dynamically choose cluster count
    group_count = min(5, len(cleaned)//2) if len(cleaned) > 2 else 1
    if group_count > 1:
        kmeans = KMeans(n_clusters=group_count, random_state=0).fit(embeddings)
        clusters = {i: [] for i in range(group_count)}
        for idx, label in enumerate(kmeans.labels_):
            clusters[label].append(cleaned[idx])
    else:
        clusters = {0: cleaned}

    # Display clusters
    st.subheader("📊 Keyword Groups")
    for gi, group in clusters.items():
        st.write(f"Group {gi+1}: {group}")

    # Fetch outlines using Wikipedia
    def fetch_outline(kw):
        try:
            wiki = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{kw}")
            if wiki.status_code == 200:
                data = wiki.json()
                summary = data.get('extract', f"Brief intro about {kw}.")
            else:
                summary = f"Introduction about {kw}."
        except:
            summary = f"Introduction about {kw}."

        headings = [
            f"What is {kw}?",
            f"Key Insights about {kw}",
            f"Applications of {kw}",
            f"Future of {kw}"
        ]
        return {"Intro": summary, "Sections": headings, "Conclusion": f"Summary for {kw}"}

    outlines = {}
    for gi, group in clusters.items():
        outlines[gi] = [fetch_outline(kw) for kw in group]

    st.subheader("📚 Generated Outlines")
    for gi, group_outline in outlines.items():
        st.write(f"Group {gi+1}:")
        for outline in group_outline:
            st.json(outline)

    # Post ideas
    def generate_post_idea(group):
        return f"Write a detailed post of comparing {', '.join(group)}, including use cases, pros, and trends."

    post_ideas = {gi: generate_post_idea(group) for gi, group in clusters.items()}
    st.subheader("💡 Post Ideas")
    st.json(post_ideas)

    # Generate PDF
    if st.button("📄 Download Keyword Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0,10,"Keyword Research Report", ln=1)
        pdf.cell(0,10, f"Input Keywords: {'; '.join(keywords)}", ln=1)
        pdf.cell(0,10, f"Cleaned Keywords: {'; '.join(cleaned)}", ln=1)
        for gi, group in clusters.items():
            pdf.cell(0,10, f"Group {gi+1}: {', '.join(group)}", ln=1)
            pdf.cell(0,10, f"Post Idea: {post_ideas[gi]}", ln=1)
            for outline in outlines[gi]:
                pdf.multi_cell(0,10, f"Outline for {', '.join(group)}:")
                for key, value in outline.items():
                    pdf.multi_cell(0,10, f"{key}: {value}")
        pdf.output("keyword_report.pdf")
        st.success("PDF saved successfully as keyword_report.pdf ✅")
else:
    st.info("Please enter or upload at least one keyword to begin.")
