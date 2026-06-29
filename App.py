import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash

import os
import re
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pdfplumber

# Core Business Logics & External Pipeline Wrappers
from jobs import get_job_recommendations, get_live_realtime_jobs, add_new_job_to_db, init_db

app = Flask(__name__)
CORS(app)
app.secret_key = "resume_parser_secret_key"
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="resume_parser_db",
        port=3306
    )

ALLOWED_EXTENSIONS = {'pdf','docx'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ================= EXTRACTION SERVICE HELPERS =================

def pdf_to_text(file_storage_obj):
    """
    Safely extracts plain-text data from a live in-memory binary file stream buffer.
    """
    text = ""
    try:
        # Rewind data pointer to zero to eliminate multi-pass state decay
        file_storage_obj.seek(0)
        with pdfplumber.open(file_storage_obj) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        app.logger.error(f"Extraction stream failed processing document: {str(e)}")
    return text


def clean_resume(txt):
    txt = txt.lower()
    txt = re.sub(r'http\S+', ' ', txt)
    txt = re.sub(r'@\S+', ' ', txt)
    txt = re.sub(r'#\S+', ' ', txt)
    txt = re.sub(r'[^\w\s]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()


def extract_email(text):
    pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    match = re.search(pattern, text)
    return match.group().strip() if match else "Not Extracted"


def extract_name(text, filename):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        potential_name = lines[0].rstrip('., ')
        if not any(char.isdigit() for char in potential_name) and len(potential_name.split()) <= 4:
            return potential_name.title()

    name_from_file = os.path.splitext(filename)[0]
    name_from_file = re.sub(r'(?i)resume|cv|updated|final|job|copy', '', name_from_file)
    name_from_file = re.sub(r'[^a-zA-Z]', ' ', name_from_file)
    return name_from_file.strip().title() if name_from_file.strip() else "Candidate Profile"


def extract_phone(text):
    pattern = r'(?:\+91[\s-]?)?[6-9]\d{2,4}[\s-]?\d{5,7}'
    match = re.search(pattern, text)
    if match:
        phone = re.sub(r'[\s-]', '', match.group())
        return phone if phone.startswith('+91') else '+91' + phone[-10:]
    return "Not Extracted"


def extract_skills(text):
    skills_list = ["Python", "Java", "SQL", "Machine Learning", "Deep Learning", "Flask",
                   "React", "HTML", "CSS", "JavaScript", "MongoDB", "Power BI", "Excel",
                   "TensorFlow", "Pandas", "NumPy", "Django", "AWS", "Docker", "Tableau"]
    return [s for s in skills_list if s.lower() in text.lower()]


def extract_education(text):
    edu_keywords = ["B.E", "BTECH", "MTECH", "SSC", "HSC", "Bachelor", "Master", "Engineering", "B.Sc", "M.Sc"]
    return list(set([e for e in edu_keywords if e.lower() in text.lower()]))


def extract_experience(text):
    exp = re.findall(r'(\d+)\s+years?', text.lower())
    return exp[0] + " Years" if exp else "Fresher"


# ================= SCORING ENGINE METRICS =================

def predict_category(text):
    text = text.lower()
    scores = {
        "AI/ML Engineer": sum(k in text for k in ["machine learning", "deep learning", "tensorflow", "nlp"]),
        "Data Analyst": sum(k in text for k in ["sql", "power bi", "tableau", "pandas", "analysis"]),
        "Full Stack Developer": sum(k in text for k in ["react", "node", "flask", "django", "javascript"]),
        "Java Developer": sum(k in text for k in ["java", "spring", "hibernate"]),
        "Cloud Engineer": sum(k in text for k in ["aws", "docker", "kubernetes", "azure"])
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Software Engineer"


def calculate_match_score(resume_text, job_description):
    if not job_description.strip():
        return 0
    important_keywords = {"python", "sql", "java", "machine learning", "excel", "tableau", "pandas", "aws", "react",
                          "flask"}
    resume_words = set(resume_text.lower().split())
    jd_words = set(job_description.lower().split())

    relevant_jd_keywords = jd_words.intersection(important_keywords)
    if not relevant_jd_keywords:
        match = resume_words.intersection(jd_words)
        return round(min((len(match) / max(len(jd_words), 1)) * 100, 100))

    match = resume_words.intersection(relevant_jd_keywords)
    return round((len(match) / len(relevant_jd_keywords)) * 100)


def calculate_ats_score(skills, education, experience, match_score):
    score = (min(len(skills) * 7, 40) + min(len(education) * 10, 20) +
             (20 if experience != "Fresher" else 10) + int(match_score * 0.2))
    return min(score, 100)


def get_keyword_analysis(resume_text, job_description):
    important_keywords = {"python", "sql", "java", "machine learning", "deep learning", "flask",
                          "react", "html", "css", "javascript", "mongodb", "power bi", "excel",
                          "tensorflow", "pandas", "numpy", "django", "aws", "docker", "tableau"}

    resume_words = set(resume_text.lower().split())
    jd_words = set(job_description.lower().split())

    target_jd_skills = jd_words.intersection(important_keywords)
    if not target_jd_skills:
        target_jd_skills = important_keywords

    matched = [s for s in target_jd_skills if s in resume_words]
    missing = [s for s in target_jd_skills if s not in resume_words]

    return [m.upper() for m in matched], [g.upper() for g in missing]


def generate_optimized_resume(name, email, phone, category, current_skills, missing_keywords, education, experience):
    all_skills = list(set(current_skills + [m.title() for m in missing_keywords]))
    skills_block = ", ".join(all_skills)

    edu_block = ", ".join(education) if education else "Bachelor of Engineering / Science in Computer Fields"
    exp_title = f"Associate {category}" if experience == "Fresher" else f"Senior {category}"
    exp_years = "0-1 Year (Entry Level / Academic Projects)" if experience == "Fresher" else experience

    return f"""=========================================================================
                                  {name.upper()}
=========================================================================
Email: {email} | Phone: {phone} | Target Role: {category}

-------------------------------------------------------------------------
PROFESSIONAL SUMMARY
-------------------------------------------------------------------------
Highly motivated and analytical professional with direct expertise in application building, 
seeking to excel as a {category}. Proven capability in system engineering, automated 
problem-solving, and utilizing production-ready keywords like {", ".join([m.title() for m in missing_keywords[:3]]) if missing_keywords else 'modern tools'}.

-------------------------------------------------------------------------
TECHNICAL REQUISITE SKILLS
-------------------------------------------------------------------------
* Core Expertise: {skills_block}
* Platforms & Frameworks: Software Engineering Lifecycle, Agile Frameworks

-------------------------------------------------------------------------
PROFESSIONAL EXPERIENCE & HIGHLIGHTED PROJECTS
-------------------------------------------------------------------------
Project Lead / Developer | {exp_title}
Duration: {exp_years}
* Spearheaded end-to-end development of data pipelines and technical applications using {all_skills[0] if all_skills else 'Python'}.
* Successfully integrated core industry methodologies involving {", ".join([m.title() for m in missing_keywords[-2:]]) if len(missing_keywords) >= 2 else 'advanced development models'} to elevate target efficiency.
* Managed clean extraction documentation and debugging, reducing runtime limits by 15%.

-------------------------------------------------------------------------
EDUCATION SUMMARY
-------------------------------------------------------------------------
* Higher Degree / Certification: {edu_block}
=========================================================================""".strip()


def interview_questions(skills):
    q_map = {
        "python": "What is the difference between a list and a tuple in Python? When would you use each?",
        "sql": "Explain the difference between a clustered and a non-clustered index in SQL.",
        "machine learning": "How do you handle missing or imbalanced data in a machine learning dataset?",
        "deep learning": "What is the vanishing gradient problem, and how do activation functions like ReLU help?",
        "flask": "How do you handle user authentication and session management in a Flask application?",
        "react": "What is the virtual DOM in React, and how does it improve performance?",
        "html": "What is semantic HTML, and why is it important for SEO and accessibility?",
        "css": "Explain the CSS Box Model and the difference between padding and margin.",
        "javascript": "What is the difference between 'let', 'const', and 'var' in modern JavaScript?",
        "mongodb": "When would you choose a NoSQL database like MongoDB over a relational SQL database?",
        "power bi": "What are DAX formulas in Power BI, and how do you optimize a slow report?",
        "excel": "How do you use VLOOKUP/XLOOKUP and Pivot Tables to clean and summarize data?",
        "tensorflow": "What is a tensor in TensorFlow, and how does a computational graph work?",
        "pandas": "How do you handle duplicate rows or merge two dataframes in Pandas?",
        "numpy": "Why are NumPy arrays faster and preferred over standard Python lists for data processing?",
        "django": "Explain the MTV (Model-Template-View) architecture pattern used in Django.",
        "aws": "What is the difference between an Amazon EC2 instance and an AWS Lambda function?",
        "docker": "What is a Docker image versus a Docker container, and how do they ensure environment consistency?",
        "tableau": "What is the difference between a live connection and an extract in Tableau?"
    }
    questions = []
    for skill in skills:
        skill_lower = skill.lower()
        if skill_lower in q_map:
            questions.append(q_map[skill_lower])

    default_qs = [
        "Walk me through the most challenging project listed on your resume.",
        "How do you stay updated with the latest tools and technologies in your domain?",
        "Can you describe a time when you had to clean a highly unorganized or messy dataset?",
        "Why are you interested in this specific role, and what makes you a good fit?"
    ]
    seen = set()
    final_questions = []
    for q in (questions + default_qs):
        if q not in seen:
            seen.add(q)
            final_questions.append(q)
    return final_questions[:5]


# ================= APP CONTROL ROUTE ENGINE =================

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/login', methods=['GET', 'POST'])
def login():

    error = None

    if request.method == 'POST':

        email = request.form.get('email')
        password = request.form.get('password')

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            "SELECT * FROM users WHERE email=%s",
            (email,)
        )

        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if user and check_password_hash(
            user['password'],
            password
        ):

            session['user'] = user['email']

            return redirect(url_for('home'))

        else:

            error = "Invalid Email or Password"

    return render_template(
        'login.html',
        error=error
    )

@app.route('/logout')
def logout():

    session.clear()

    flash(
        "Logged out successfully",
        "success"
    )

    return redirect(
        url_for('login')
    )

@app.route('/signup', methods=['GET', 'POST'])
def signup():

    error = None

    if request.method == 'POST':

        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:

            error = "Passwords do not match"

            return render_template(
                'signup.html',
                error=error
            )

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE email=%s",
            (email,)
        )

        existing_user = cursor.fetchone()

        if existing_user:

            error = "Email already registered"

        else:

            hashed_password = generate_password_hash(password)

            cursor.execute(
                """
                INSERT INTO users
                (email, password)
                VALUES (%s, %s)
                """,
                (
                    email,
                    hashed_password
                )
            )

            conn.commit()

            cursor.close()
            conn.close()

            flash(
                "Registration Successful. Please Login.",
                "success"
            )

            return redirect(url_for('login'))

        cursor.close()
        conn.close()

    return render_template(
        'signup.html',
        error=error
    )


@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    files = request.files.getlist('resume')
    job_description = request.form.get('job_description', "")
    results = []

    for idx, file in enumerate(files):
        if not file or not file.filename:
            continue

        if not allowed_file(file.filename):
            flash(f"Unsupported format skipped: {file.filename}. Upload PDF files only.", "warning")
            continue

        filename = secure_filename(file.filename)
        raw_text = pdf_to_text(file)

        if not raw_text.strip():
            flash(f"Extraction failed for file: {filename}. Ensure it contains readable text layers.", "danger")
            continue

        cleaned_text = clean_resume(raw_text)

        skills = extract_skills(cleaned_text)
        edu = extract_education(raw_text)
        exp = extract_experience(raw_text)

        match_score = calculate_match_score(cleaned_text, job_description)
        ats_score = calculate_ats_score(skills, edu, exp, match_score)
        matched_kws, missing_kws = get_keyword_analysis(cleaned_text, job_description)

        if not job_description.strip():
            status = "Shortlisted" if ats_score >= 65 else "Rejected"
        else:
            status = "Shortlisted" if (ats_score >= 60 and match_score >= 30) else "Rejected"

        name = extract_name(raw_text, filename)
        email = extract_email(raw_text)
        phone = extract_phone(raw_text)
        category = predict_category(cleaned_text)

        suggested_resume_txt = generate_optimized_resume(
            name, email, phone, category, skills, missing_kws, edu, exp
        )

        # LOCALIZATION MIDDLEWARE: Explicitly filters out international nodes
        # Calls the dynamic job matrix inside jobs.py mapped for Indian tech hubs
        recommended_jobs = get_live_realtime_jobs(category, skills, limit=3)

        results.append({
            "id": idx,
            "name": name,
            "email": email,
            "phone": phone,
            "category": category,
            "skills": skills,
            "education": edu,
            "experience": exp,
            "match_score": match_score,
            "ats_score": ats_score,
            "interview_questions": interview_questions(skills),
            "status": status,
            "matched_keywords": matched_kws,
            "missing_keywords": missing_kws,
            "suggested_resume": suggested_resume_txt,
            "job": category,
            "recommended_jobs": recommended_jobs
        })

    session['results'] = results
    return render_template("result.html", results=results)


@app.route('/view-results')
def view_results():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("result.html", results=session.get('results', []))


@app.route('/candidate/<int:index>')
def candidate(index):
    if 'user' not in session:
        return redirect(url_for('login'))

    results = session.get('results', [])
    c = next((cand for cand in results if cand.get('id') == index), None)

    if c:
        # Fetch jobs based on candidate's skills
        # Ensure 'get_job_recommendations' takes a list of skills
        jobs = get_job_recommendations(c.get('skills', []), limit=3)
        return render_template("candidate.html", c=c, jobs=jobs)

    return "Candidate profile index not found", 404
@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        app.logger.info(f"Contact submission from {name} ({email}): {subject} - {message}")
        flash("Thank you! Your message has been sent successfully.", "success")
        return redirect(url_for('contact'))
    return render_template("contact.html")


# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         flash("Registration is currently managed by systems administration.", "info")
#         return redirect(url_for('login'))
#     return render_template("signup.html")


@app.route('/recommendations')
def recommendations():
    if 'user' not in session:
        return redirect(url_for('login'))

    uploaded_results = session.get('results', [])

    if not uploaded_results:
        flash("Please upload resumes first to initialize your profile context dashboard.", "warning")
        return redirect(url_for('home'))

    profile_recommendations = []
    seen_candidates = set()

    for candidate_data in uploaded_results:
        candidate_name = candidate_data['name']

        if candidate_name in seen_candidates:
            continue
        seen_candidates.add(candidate_name)

        domain_profile = candidate_data.get('category', 'Software Engineer')
        candidate_skills = candidate_data.get('skills', [])

        # Standardizing output channels for domestic matching
        jobs_list = get_live_realtime_jobs(domain_profile, candidate_skills, limit=3)

        profile_recommendations.append({
            "candidate_name": candidate_name,
            "profile_name": domain_profile,
            "jobs": jobs_list
        })

    return render_template('recommendations.html', profiles=profile_recommendations)


@app.route('/hr/add-job', methods=['GET', 'POST'])
def hr_add_job():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        title = request.form.get('title')
        company = request.form.get('company')
        profile = request.form.get('profile')
        location = request.form.get('location')
        salary = request.form.get('salary')
        skills = request.form.get('skills')

        add_new_job_to_db(title, company, profile, location, salary, skills)
        flash(f"Job opening for '{title}' posted successfully into tracking pipeline!", "success")
        return redirect(url_for('view_results'))

    return render_template("hr_post_job.html")



if __name__ == "__main__":
    init_db()
    app.run(debug=True)