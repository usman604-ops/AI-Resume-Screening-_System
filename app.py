from flask import Flask, render_template, request
import os
import PyPDF2
import docx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = "resumes"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def extract_pdf(file):

    text = ""

    with open(file, "rb") as f:
        reader = PyPDF2.PdfReader(f)

        for page in reader.pages:
            text += page.extract_text()

    return text


def extract_docx(file):

    doc = docx.Document(file)

    text = ""

    for para in doc.paragraphs:
        text += para.text

    return text


def match_resume(resume_text, job_desc):

    text = [resume_text, job_desc]

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(text)

    similarity = cosine_similarity(vectors[0], vectors[1])

    score = similarity[0][0]

    return round(score * 100, 2)


@app.route("/", methods=["GET", "POST"])

def index():

    score = None

    if request.method == "POST":

        job_desc = request.form["jobdesc"]

        file = request.files["resume"]

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

        file.save(filepath)

        if file.filename.endswith(".pdf"):
            resume_text = extract_pdf(filepath)

        elif file.filename.endswith(".docx"):
            resume_text = extract_docx(filepath)

        else:
            resume_text = ""

        score = match_resume(resume_text, job_desc)

    return render_template("index.html", score=score)


if __name__ == "__main__":
    app.run(debug=True)
