from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import re
import pickle

# ===================== FLASK APP =====================
app = Flask(__name__)

# ===================== LOAD MODELS =====================
rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
rf_classifier_job_recommendation = pickle.load(open('models/rf_classifier_job_recommendation.pkl', 'rb'))
tfidf_vectorizer_job_recommendation = pickle.load(open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb'))

# ===================== CLEAN RESUME =====================
def cleanResume(txt):
    txt = re.sub(r'http\S+\s', ' ', txt)
    txt = re.sub(r'RT|cc', ' ', txt)
    txt = re.sub(r'#\S+\s', ' ', txt)
    txt = re.sub(r'@\S+', ' ', txt)
    txt = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

# ===================== NORMALIZE TEXT =====================
def normalize_text(text):
    text = re.sub(r'[\n\r]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ===================== PREDICTIONS =====================
def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    return rf_classifier_categorization.predict(resume_tfidf)[0]

def job_recommendation(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    return rf_classifier_job_recommendation.predict(resume_tfidf)[0]

# ===================== PDF TO TEXT =====================
def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# ===================== RESUME PARSING =====================
def extract_contact_number_from_resume(text):
    pattern = r'\+?\d[\d\s.-]{8,}\d'
    match = re.search(pattern, text)
    return match.group() if match else "Not Found"

def extract_email_from_resume(text):
    text = text.replace('(at)', '@').replace('[at]', '@').replace('{at}', '@')
    text = text.replace(' ', '')
    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    match = re.search(pattern, text)
    return match.group() if match else "Not Found"


def extract_name_from_resume(text):
    """
    Extract the first name from resume using simple regex heuristics.
    - Scans first 50 lines.
    - Skips lines with headings like Project, Skills, etc.
    - Returns the first capitalized word in a likely name line.
    """
    lines = text.split('\n')[:50]  # first 50 lines
    skip_keywords = ['resume', 'curriculum vitae', 'cv', 'profile', 'summary',
                     'project', 'company', 'experience', 'education', 'skills', 'objective']

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        if any(word.lower() in line_clean.lower() for word in skip_keywords):
            continue

        # Remove punctuation
        line_clean = re.sub(r'[:\-|,]', ' ', line_clean)
        line_clean = re.sub(r'\s+', ' ', line_clean)

        # Look for first capitalized word
        match = re.match(r'\b([A-Z][a-z]+)\b', line_clean)
        if match:
            return match.group(1)  # return first name

    return "Not Found"



def extract_skills_from_resume(text):
    text = normalize_text(text)
    skills_list = [
        'Python', 'SQL', 'Machine Learning', 'Deep Learning', 'Data Analysis', 'Data Science',
        'Flask', 'Django', 'HTML', 'CSS', 'JavaScript', 'React', 'Git',
        'Power BI', 'Tableau', 'Pandas', 'Numpy', 'Scikit-learn', 'TensorFlow', 'Keras'
    ]
    skills = []
    for skill in skills_list:
        pattern = rf"\b{re.escape(skill).replace(r'\ ', r'[-\s]?')}\b"
        if re.search(pattern, text, re.IGNORECASE):
            skills.append(skill)
    return skills if skills else ["No skills found"]

def extract_education_from_resume(text):
    text = normalize_text(text)
    education_keywords = [
        'Computer Science', 'Information Technology', 'Data Science', 'Computer Engineering',
        'Bachelor', 'Master', 'B.Tech', 'M.Tech', 'B.E', 'M.E'
    ]
    education = []
    for edu in education_keywords:
        pattern = rf"\b{re.escape(edu).replace(r'\ ', r'[-\s]?')}\b"
        if re.search(pattern, text, re.IGNORECASE):
            education.append(edu)
    return education if education else ["No education found"]

# ===================== ROUTES =====================
@app.route('/')
def resume():
    return render_template("resume.html")

@app.route('/pred', methods=['POST'])
def pred():
    if 'resume' not in request.files:
        return render_template("resume.html", message="No resume file uploaded.")

    file = request.files['resume']
    filename = file.filename.lower()

    if filename.endswith('.pdf'):
        text = pdf_to_text(file)
    elif filename.endswith('.txt'):
        text = file.read().decode('utf-8')
    else:
        return render_template("resume.html", message="Invalid file format. Upload PDF or TXT.")

    predicted_category = predict_category(text)
    recommended_job = job_recommendation(text)
    phone = extract_contact_number_from_resume(text)
    email = extract_email_from_resume(text)
    name = extract_name_from_resume(text)
    extracted_skills = extract_skills_from_resume(text)
    extracted_education = extract_education_from_resume(text)

    return render_template(
        "resume.html",
        predicted_category=predicted_category,
        recommended_job=recommended_job,
        phone=phone,
        email=email,
        name=name,
        extracted_skills=extracted_skills,
        extracted_education=extracted_education
    )

# ===================== MAIN =====================
if __name__ == '__main__':
    app.run(debug=True)
