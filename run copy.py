from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import os
import uuid
import hashlib
from werkzeug.utils import secure_filename
from flask import render_template
from dotenv import load_dotenv
import pymysql

from app.services.ai_engine.score import (
    detect_grammar_fixes_only,
    detect_minor_edits,
    detect_structural_changes,
    detect_major_rewrite,
    detect_behavioral_inconsistency
)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'app', 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'app', 'static')
app = Flask(
    __name__, 
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR ,
)

load_dotenv()
app.secret_key = 'prufia_user' 
# Configuration
app.config['BASELINE_FOLDER'] = 'baseline'
app.config['ASSIGNMENT_FOLDER'] = 'assignments'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx'}  # Allowed file extensions

# Ensure directories exist
os.makedirs(app.config['BASELINE_FOLDER'], exist_ok=True)
os.makedirs(app.config['ASSIGNMENT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

    
def db_connection():    
    try:
        conn = pymysql.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME'),
            port=3306,
            # auth_plugin='mysql_native_password'
        )
        return conn
    except pymysql.MySQLError as e:
        print(f"Error connecting to MySQL: {e}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/student-login')
def studentlogin():    
    return render_template('studentlogin.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('studentlogin'))


@app.route('/login', methods=['POST'])
def login():
    name = request.form['name']
    code = request.form['code']
    conn = None
    try:
        conn = db_connection()
        with conn.cursor() as cursor:
            hashed_code = hashlib.sha256(code.encode()).hexdigest()    
            cursor.execute("SELECT * FROM students WHERE name_or_alias=%s AND password_hash=%s", (name, hashed_code))
            student = cursor.fetchone()  # Use fetchone() instead of fetchall()
            
            if student:  # If a student was found
                session['student_id'] = student[0]
                session['student_name'] = student[1]
                return redirect(url_for('student'))
            
            # If no student found
            return render_template('studentlogin.html', error='Invalid credentials')
            
    except Exception as e:
        print(f"Login error: {e}")
        return render_template('studentlogin.html', error='An error occurred during login')
        
    finally:
        if conn:  # Only close connection if it was created
            conn.close()
    


@app.route('/teacher-login')
def teacherlogin():
    return render_template('teacherlogin.html')

@app.route('/student')
def student():
    if 'student_id' not in session:
        return redirect(url_for('studentlogin'))
    
    student_id = session['student_id']
    conn = db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM submissions WHERE student_id=%s ORDER BY created_at DESC",
                (student_id,)
            )
            baselines = cursor.fetchall()  # Get all records
            
    except pymysql.MySQLError as e:
        conn.rollback()
        print(f"Database error: {str(e)}")  # Log the error
        baselines = []
    finally:
        conn.close()
    return render_template(
        'student.html', 
        student_name=session['student_name'],
        baselines=baselines  # Pass all baselines to template
    )


@app.route('/teacher')
def teacher():
    conn = db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(
                "SELECT   s.id, s.student_id, st.name_or_alias,   s.baseline_1_path , s.baseline_2_path , s.created_at, s.submission_path, s.score_baseline_1, s.score_baseline_2, s.final_score, s.trust_flag, s.interpretation FROM submissions s JOIN students st ON s.student_id = st.id ORDER BY s.created_at DESC"
            )
            baselines = cursor.fetchall()  # Get all records
            
    except pymysql.MySQLError as e:
        conn.rollback()
        print(f"Database error: {str(e)}")  # Log the error
        baselines = []
    finally:
        conn.close()
    return render_template(
        'teacher.html',
        baselines=baselines 
    )


@app.route('/analyze', methods=['POST']) 
def analyze():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data received"
            }), 400
        tech = data.get('tech')
        extension = data.get('extension')
        baseline_1 = data.get('baseline_1')
        baseline_2 = data.get('baseline_2')
        student_id = data.get('studentId')

        if not all([tech, extension, baseline_1, baseline_2, student_id]):
            return jsonify({
                "status": "error",
                "message": "Missing required fields"
            }), 400
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        baseline_dir = os.path.join(base_dir, 'baseline', student_id)
        teacher_dir = os.path.join(base_dir, 'assignments')
        
        # Corrected file paths
        baseline_1_path = os.path.join(baseline_dir, f"{baseline_1}.txt")
        baseline_2_path = os.path.join(baseline_dir, f"{baseline_2}.txt")
        teacher_path = os.path.join(teacher_dir, f"{tech}")  # Fixed

        try:
            with open(baseline_1_path, 'r') as f1, \
                 open(baseline_2_path, 'r') as f2, \
                 open(teacher_path, 'r') as f3:
                
                baseline_1_content = f1.read()
                baseline_2_content = f2.read()
                teacher_content = f3.read()
                
                gmark1=detect_grammar_fixes_only(teacher_content,baseline_1_content)
                gmark2=detect_grammar_fixes_only(teacher_content,baseline_2_content)
                gmark=round((gmark1 + gmark2) / 2); 

                rmark1=detect_minor_edits(teacher_content,baseline_1_content)
                rmark2=detect_minor_edits(teacher_content,baseline_2_content)
                minor=round((rmark1 + rmark2) / 2); 


                smark1=detect_structural_changes(teacher_content,baseline_1_content)
                smark2=detect_structural_changes(teacher_content,baseline_2_content)
                structural=round((smark1 + smark2) / 2); 


                mmark1=detect_major_rewrite(teacher_content,baseline_1_content)
                mmark2=detect_major_rewrite(teacher_content,baseline_2_content)
                major=round((mmark1 + mmark2) / 2); 



                results={
                    "major":major,
                    "structural":structural,
                    "minor":minor,
                    "grammar":gmark
                }

                weights = {
                    "grammar": 0.95,
                    "minor": 0.85,
                    "structural": 0.62,
                    "major": 0.30
                }

                active_scores = {key: weights[key] for key in results if results[key] == 1}
                
                if results["major"]:  # If major rewrite was detected
                    final_score = min(active_scores.values())  # Take the lowest score
                else:
                    # Average the scores if no major rewrite
                    final_score = sum(active_scores.values()) / len(active_scores) if active_scores else 1.0

                # Determine status
                if final_score >= 0.9:
                    status = "Authorship Confirmed"
                    flag = "green"
                elif final_score >= 0.7:
                    status = "Likely Match"
                    flag = "yellow"
                elif final_score >= 0.5:
                    status = "Authorship Doubtful"
                    flag = "orange"
                else:
                    status = "Authorship Mismatch"
                    flag = "red"

                # Prepare the output
                return jsonify({
                    "status": "success",
                    "score": int(final_score * 100),
                    "status": status,
                    "flag": flag,
                    "active_detections": list(active_scores.keys()),
                    "explanation": f"Detected: {', '.join(active_scores.keys())}",
                    "message": "Files read successfully"
                }), 200

                # bmark1=detect_behavioral_inconsistency(teacher_content,baseline_1_content)
                # bmark2=detect_behavioral_inconsistency(teacher_content,baseline_2_content)
                # behavioral=round((bmark1 + bmark2) / 2); 

                
                
        except FileNotFoundError as e:
            return jsonify({
                "status": "error",
                "message": f"File not found: {str(e)}. Path: {teacher_path}"  # More detailed error
            }), 404
        except IOError as e:
            return jsonify({
                "status": "error", 
                "message": f"Error reading files: {str(e)}"
            }), 500

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Server error: {str(e)}"
        }), 500
    
@app.route('/submit_baseline', methods=['POST'])
def submit_baseline():
    """Endpoint to submit baseline writing samples for a student"""
    try:
        # Check if user is logged in
        if 'student_id' not in session:
            return jsonify({"error": "You must be logged in to submit baselines"}), 401

        # Get student_id from session
        student_id = str(session['student_id'])  # Convert to string if it's not already
        student_name = session.get('student_name', 'Unknown')

        # Get form data
        p1 = request.form.get('prompt1', '')

        # typing_metrics = request.form.get('typing_metrics')
        # print("typing_metrics===>", typing_metrics)


        p2 = request.form.get('prompt2', '')
        
        if not p1 or not p2:
            return jsonify({"error": "Both prompts are required"}), 400

        # Create student folder
        folder = os.path.join(app.config['BASELINE_FOLDER'], student_id)
        os.makedirs(folder, exist_ok=True)
        
        # Generate unique filenames
        uuid_obj = uuid.uuid4()
        first_part = uuid_obj.hex[:8]      # "5e2a1155"
        second_part = uuid_obj.hex[8:12]   # "5a6e"
        result = f"{first_part}-{second_part}"

        p1_filename = f"{result}.txt"
        
        uuid_obj = uuid.uuid4()
        first_part = uuid_obj.hex[:8]      # "5e2a1155"
        second_part = uuid_obj.hex[8:12]   # "5a6e"
        result = f"{first_part}-{second_part}"
        p2_filename = f"{result}.txt"
        
        # Save prompts
        p1_path = os.path.join(folder, p1_filename)
        p2_path = os.path.join(folder, p2_filename)
        
        with open(p1_path, 'w') as f:
            f.write(p1)
        with open(p2_path, 'w') as f:
            f.write(p2)

        # Database operations
        conn = db_connection()
        try:
            with conn.cursor() as cursor:
                # Insert first baseline sample
                cursor.execute(
                    "INSERT INTO submissions (student_id, baseline_1_path, baseline_2_path, semester_id) VALUES (%s, %s, %s, %s)",
                    (student_id, p1_path, p2_path,"2025-Pilot")
                )  
                print("insert success!")
                conn.commit()
                
        except pymysql.MySQLError as e:
            conn.rollback()
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        finally:
            conn.close()
            
        return jsonify({
            "status": "success",
            "message": f"Baseline saved for {student_name}",
            "student_id": student_id,
            "file_paths": {
                "prompt1": p1_path,
                "prompt2": p2_path
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_assignments', methods=['POST'])
def upload_assignments():
    """Endpoint to upload assignments and automatically match with baselines"""
    results = {}
    
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
        
    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No selected files"}), 400
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            prefix = filename.split('_')[0]
            
            # Find best matching student (mock implementation)
            match = find_best_match(prefix, app.config['BASELINE_FOLDER'])
            
            # Save to appropriate folder
            folder_path = os.path.join(
                app.config['ASSIGNMENT_FOLDER']
            )
            os.makedirs(folder_path, exist_ok=True)
            
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(folder_path, unique_filename)
            file.save(file_path)
            
            # Calculate score if matched
            if match:
                baseline_path = os.path.join(app.config['BASELINE_FOLDER'], match)
                score = calculate_combined_score(baseline_path, file_path)
                
                # Determine interpretation
                if score >= 0.85:
                    label = "Strong Match"
                elif score >= 0.70:
                    label = "Moderate Match"
                elif score >= 0.50:
                    label = "Weak Match"
                else:
                    label = "Very Low Match"
                    
                results[filename] = {
                    'student_id': match,
                    'score': round(score, 4),
                    'interpretation': label,
                    'saved_path': file_path
                }
            else:
                results = {
                    'student_id': 'Unmatched',
                    'score': 'N/A',
                    'interpretation': 'Manual Match Required',
                    'saved_path': file_path
                }
        else:
            results[file.filename] = {
                'error': 'Invalid file type',
                'allowed_types': list(app.config['ALLOWED_EXTENSIONS'])
            }
    
    return jsonify(results)

@app.route('/manual_match', methods=['POST'])
def manual_match():
    """Endpoint for manually matching assignments to students"""
    try:
        name = request.form.get('file_name')
        sid = request.form.get('student_id')
        
        if not name or not sid:
            return jsonify({"error": "Both file_name and student_id are required"}), 400
            
        unmatched_path = os.path.join(app.config['ASSIGNMENT_FOLDER'])
        file_path = None
        
        for root, dirs, files in os.walk(unmatched_path):
            if name in files:
                file_path = os.path.join(root, name)
                break
        if not file_path:
            return jsonify({"error": "File not found in unmatched directory"}), 404
        base_path = os.path.join(app.config['BASELINE_FOLDER'], sid)
        if not os.path.exists(base_path):
            return jsonify({"error": "Student baseline not found"}), 404
        student_folder = os.path.join(app.config['ASSIGNMENT_FOLDER'], sid)
        os.makedirs(student_folder, exist_ok=True)
        new_path = os.path.join(student_folder, name)
        os.rename(file_path, new_path)
        score = calculate_combined_score(base_path, new_path)
        
        return jsonify({
            'status': 'success',
            'student_id': sid,
            'score': round(score, 4),
            'file_path': new_path
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Mock implementations for the scoring and matching functions
def calculate_combined_score(baseline_path, submission_path):
    """Calculate a combined similarity score between baseline and submission"""
    # In a real implementation, this would compare writing styles
    # For now, return a mock score between 0 and 1
    return 100*round(0.5 + (hash(submission_path) % 5000) / 10000, 4)

def find_best_match(prefix, baseline_folder):
    """Find the best matching student ID for a given prefix"""
    # In a real implementation, this would use fuzzy matching
    # For now, just return the first student ID that starts with the prefix
    try:
        for student_id in os.listdir(baseline_folder):
            if student_id.startswith(prefix):
                return student_id
    except FileNotFoundError:
        pass
    return None

def run_server():
    try:
        port = int(os.environ.get('PORT', 5000))
        host = os.environ.get('HOST', '0.0.0.0')
        
        # Try multiple ports if default is unavailable
        for p in [port, port + 1, port + 2]:
            try:
                app.run(host=host, port=p, debug=True)
                break
            except OSError as e:
                if "address in use" in str(e):
                    print(f"Port {p} in use, trying next...")
                    continue
                raise
    except Exception as e:
        print(f"Server failed: {str(e)}")
        # Additional error handling/logging here

if __name__ == '__main__':     
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # This prevents the socket error
    )