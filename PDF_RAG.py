    import ollama
    from sentence_transformers import SentenceTransformer, util
    from sklearn.metrics.pairwise import cosine_similarity
    from PyPDF2 import PdfReader
    import os

    model = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_text_from_pdf(pdf_path):
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    pdf_path = "C:\Users\HP\Downloads\MongoDB-The-Definitive-Guide-2nd-Edition.pdf"

    pdf_text = extract_text_from_pdf(pdf_path)
    print("Length:", len(pdf_text))
    print(pdf_text[:500])

    def get_vect(text):
    return model.encode(text)

    #Step 2 — Chunk the Text (VERY IMPORTANT)
    #We don’t embed whole PDF at once ❌
    #We split into smaller pieces (chunks) ✅

    def chunk_text(text, chunk_size=500, overlap=100):
        if not text or len(text.strip()) == 0:
            print("No text found to chunk.")
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    chunks = chunk_text(pdf_text)

    print(f"Number of chunks: {len(chunks)}")
    print(chunks[0])

    print("Length of extracted text:", len(pdf_text))
    print(pdf_text[:500])


    documents = chunks

    print(len(documents))
    print(documents[0])


    doc_emb = get_vect(documents)
    print(doc_emb.shape)


    def retrieve(query, top_k=1):
        query_emb = model.encode(query)

        similarities = cosine_similarity(
            query_emb.reshape(1, -1),
            doc_emb
        )[0]

        top_indices = similarities.argsort()[-top_k:][::-1]

        return [documents[i] for i in top_indices]

    results = retrieve("What are the three specific restrictions for naming a MongoDB collection?", top_k=2)

    for r in results:
        print("-----")
        print(r)