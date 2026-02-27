from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import re
import pickle
import os

app = Flask(__name__)

# ===================== LOAD MODELS =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rf_classifier_categorization = pickle.load(open(os.path.join(BASE_DIR,'models/rf_classifier_categorization.pkl'),'rb'))
tfidf_vectorizer_categorization = pickle.load(open(os.path.join(BASE_DIR,'models/tfidf_vectorizer_categorization.pkl'),'rb'))

rf_classifier_job_recommendation = pickle.load(open(os.path.join(BASE_DIR,'models/rf_classifier_job_recommendation.pkl'),'rb'))
tfidf_vectorizer_job_recommendation = pickle.load(open(os.path.join(BASE_DIR,'models/tfidf_vectorizer_job_recommendation.pkl'),'rb'))

# ===================== PDF TO TEXT =====================
def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

# ===================== CLEAN RESUME =====================
def cleanResume(txt):
    txt = re.sub(r'http\S+\s*', ' ', txt)
    txt = re.sub(r'RT|cc', ' ', txt)
    txt = re.sub(r'#\S+', ' ', txt)
    txt = re.sub(r'@\S+', ' ', txt)
    txt = re.sub(r'[^\w\s]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

# ===================== NORMALIZE TEXT =====================
def normalize_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ===================== PREDICTIONS =====================
def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    vector = tfidf_vectorizer_categorization.transform([resume_text])
    return rf_classifier_categorization.predict(vector)[0]

def job_recommendation(resume_text):
    resume_text = cleanResume(resume_text)
    vector = tfidf_vectorizer_job_recommendation.transform([resume_text])
    return rf_classifier_job_recommendation.predict(vector)[0]

# ===================== EMAIL EXTRACTION =====================
def extract_email_from_resume(text):

    text = re.sub(r'\s+', '', text)

    match = re.search(
        r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(?:com|in|org|edu|net)',
        text
    )

    return match.group(0) if match else "Not Found"

# ===================== PHONE EXTRACTION =====================
def extract_contact_number_from_resume(text):

    phones = re.findall(r'\+?\d[\d\s\-]{8,}\d', text)

    if phones:
        phone = max(phones, key=len)
        phone = re.sub(r'\s+', '', phone)
        return phone

    return "Not Found"

# ===================== NAME EXTRACTION =====================
def extract_name_from_resume(text):

    lines = text.split('\n')[:20]

    blacklist = {
        'python','java','javascript','html','css','react','node','sql',
        'mysql','mongodb','bootstrap','vite','restful','api','apis',
        'database','databases','skills','education','projects',
        'experience','profile','summary','objective','contact',
        'languages','frameworks','tools','concepts'
    }

    for line in lines:

        line = line.strip()

        if not line:
            continue

        clean = re.sub(r'[^A-Za-z\s]', ' ', line)
        words = clean.split()

        if 2 <= len(words) <= 3:

            valid = True

            for w in words:

                if not w[0].isupper():
                    valid = False

                if w.lower() in blacklist:
                    valid = False

            if valid:
                return " ".join(words)

    # fallback from email
    email = extract_email_from_resume(text)

    if email != "Not Found":
        name_part = email.split("@")[0]
        name_part = re.sub(r'\d+', '', name_part)
        name_part = name_part.replace(".", " ").replace("_", " ")
        return name_part.title()

    return "Not Found"

# ===================== SKILLS =====================
def extract_skills_from_resume(text):

    text = normalize_text(text)

    skills_list = [
        'Python','SQL','Machine Learning','Deep Learning','Data Analysis','Data Science',
        'Flask','Django','HTML','CSS','JavaScript','React','Git',
        'Power BI','Tableau','Pandas','Numpy','Scikit-learn','TensorFlow','Keras'
    ]

    found = []

    for skill in skills_list:
        if re.search(rf"\b{skill}\b", text, re.IGNORECASE):
            found.append(skill)

    return found if found else ["No skills found"]

# ===================== EDUCATION =====================
def extract_education_from_resume(text):

    text = normalize_text(text)

    education_keywords = [
        'Computer Science','Information Technology','Data Science','Computer Engineering',
        'Bachelor','Master','B.Tech','M.Tech','B.E','M.E'
    ]

    found = []

    for edu in education_keywords:
        if re.search(rf"\b{edu}\b", text, re.IGNORECASE):
            found.append(edu)

    return found if found else ["No education found"]

# ===================== ROUTES =====================
@app.route('/')
def home():
    return render_template("resume.html")

@app.route('/pred', methods=['POST'])
def pred():

    if 'resume' not in request.files:
        return render_template("resume.html", message="No resume file uploaded")

    file = request.files['resume']
    filename = file.filename.lower()

    if filename.endswith('.pdf'):
        text = pdf_to_text(file)

    elif filename.endswith('.txt'):
        text = file.read().decode('utf-8', errors='ignore')

    else:
        return render_template("resume.html", message="Upload PDF or TXT only")

    category = predict_category(text)
    job = job_recommendation(text)

    return render_template(
        "resume.html",
        predicted_category=category,
        recommended_job=job,
        phone=extract_contact_number_from_resume(text),
        email=extract_email_from_resume(text),
        name=extract_name_from_resume(text),
        extracted_skills=extract_skills_from_resume(text),
        extracted_education=extract_education_from_resume(text)
    )

# ===================== MAIN =====================
if __name__ == '__main__':
    app.run(debug=True)