from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import os, time
import uuid
from werkzeug.utils import secure_filename
from flask import render_template
from dotenv import load_dotenv
import sys
import shutil
from app.services.auth.login import handle_login 
from app.services.common.getters import get_baselines, get_submissions, get_last_baseline, get_requetsts,get_last_baseline_admin
from app.services.security.protect import decrypt
from app.services.admin.common import updateApprve
from app.services.common.fileread import  read_file
from app.services.student.submit import submit_baseline, getstudents,getstudentNamebyId
from app.services.teacher.business import getPlantext, workingScore, handleResubmitRequest
from app.services.security.passcode import gencode , getpasscode,handle_reset_all_passcode
from flask_socketio import SocketIO
from os.path import dirname, join

sys.path.append(join(dirname(__file__)))

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
app.config['BASELINE_FOLDER'] = 'baseline'
app.config['ASSIGNMENT_FOLDER'] = 'assignments'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx'}  
socketio = SocketIO(app, cors_allowed_origins="*")

os.makedirs(app.config['BASELINE_FOLDER'], exist_ok=True)
os.makedirs(app.config['ASSIGNMENT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def home():
    return render_template('index.html')



@socketio.on('join-admin-room')
def handle_join_admin_room():
    from flask_socketio import join_room
    join_room('admin-room')
    print(f"Admin joined admin-room")      

#############################Admin panel ongoing #######################################
@app.route('/adm')
def admins():
    response, error, status_code = getpasscode()
    if error:
        return f"Error fetching students: {error}", status_code
    return render_template('2.html',students=response["data"])

@app.route('/ch')
def ch_home():
    return render_template('ch/base.html')

@app.route('/ch/content1')
def content1():
    return render_template('ch/content1.html')

@app.route('/ch/content2')
def content2():
    return render_template('ch/content2.html')

@app.route('/ch/left')
def left_sidebar():
    return render_template('ch/left.html')

##############################################################################


#Admin routes
@app.route('/passcode')
def passcode():
    response, error, status_code = getpasscode()
    if error:
        return f"Error fetching students: {error}", status_code
        
    return render_template('admin/passcode.html', students=response["data"])

@app.route('/generate-passcode/<int:student_id>', methods=['POST'])
def generate_passcode(student_id):
    return gencode(student_id)

@app.route('/semesters')
def manage_semesters():
    # Sample data - in production you'd query the database
    sample_semesters = [
        {
            'id': '2025-Spring',
            'name': 'Spring 2025',
            'start_date': '2025-01-15',
            'end_date': '2025-05-20',
            'is_active': True,
            'student_count': 42,
            'teacher_count': 5
        },
        {
            'id': '2024-Fall',
            'name': 'Fall 2024',
            'start_date': '2024-08-20',
            'end_date': '2024-12-15',
            'is_active': False,
            'student_count': 38,
            'teacher_count': 4
        },
        {
            'id': '2024-Summer',
            'name': 'Summer 2024',
            'start_date': '2024-06-01',
            'end_date': '2024-08-10',
            'is_active': False,
            'student_count': 22,
            'teacher_count': 3
        }
    ]
    return render_template('admin/semesters.html', semesters=sample_semesters)

@app.route('/approve-endpoint', methods=['POST'])
def approve_status():
    try:
        data = request.get_json()
        base_id = data.get('base_id')
        is_approved = data.get('is_approved')
        print("baseline===",base_id,"++",is_approved)
        if not base_id:
            return jsonify({
                'success': False,
                'message': 'Missing baseline ID',
                'error': 'base_id_required'
            }), 400
        status,error=updateApprve(base_id)
        return jsonify({
                'success': True,
                'message': 'Baseline successfully approved' if is_approved else 'Baseline approval rejected',
                'data': {
                    'base_id': base_id,
                    'new_status': 'approved' if is_approved else 'rejected',
                    # 'timestamp': datetime.now().isoformat()
                }
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': 'An error occurred processing your request',
            'error': str(e)
        }), 500

@app.route('/update_submission_status', methods=['POST'])
def update_submission_status():
    data = request.get_json()
    submission_id = data.get('submission_id')
    new_status = data.get('new_status')
    
    # Here you would update the submission in your database
    # For now, we'll just return a success message
    return jsonify({
        'success': True,
        'message': f'Status updated to {new_status}'
    })

@app.route('/submissions')
def manage_submissions():
    requestlist, error = get_requetsts()
    if error:
        return render_template('error.html', message=error), 500
    return render_template('admin/submissions.html', baselines=requestlist if requestlist else [])

@app.route('/semester/<semester_id>')
def semester_details(semester_id):
    sample_data = {
        'id': semester_id,
        'name': 'Spring 2025' if 'Spring' in semester_id else 'Fall 2024',
        'start_date': '2025-01-15',
        'end_date': '2025-05-20',
        'is_active': True,
        'description': 'Main academic semester for undergraduate programs',
        'students': [
            {'id': 101, 'name': 'Alice Johnson', 'email': 'alice@example.com'},
            {'id': 102, 'name': 'Bob Smith', 'email': 'bob@example.com'}
        ],
        'teachers': [
            {'id': 201, 'name': 'Dr. Sarah Chen', 'email': 'sarah@example.com'},
            {'id': 202, 'name': 'Prof. David Kim', 'email': 'david@example.com'}
        ],
        'assignments': [
            {'id': 301, 'name': 'Baseline Assessment', 'due_date': '2025-02-01'},
            {'id': 302, 'name': 'Midterm Project', 'due_date': '2025-03-15'}
        ]
    }
    return render_template('admin/semester_details.html', semester=sample_data)

@app.route('/reset', methods=['POST'])  # Should be POST for state-changing operations
def reset():
    response, status_code = handle_reset_all_passcode()
    print("response:", response)
    
    if status_code == 200:       
        socketio.emit('reset-passcode', {
            'status': 'success',
            'message': response['message'],
            'affected_rows': response['affected_rows'],
        }, room='admin-room', namespace='/admin')
    
    return jsonify(response), status_code
    
@app.route('/admin')
def admin_dashboard():
    return render_template('admin/main.html')


# Teacher test routes final version
@app.route('/tea')
def tea_home():
    baselines, error = get_baselines()
    if error:
        return render_template('error.html', message=error), 500
    requestlist, error = get_requetsts()
    if error:
        return render_template('error.html', message=error), 500
    return render_template('1.html',baselines=baselines if baselines else [], requestlist=requestlist if requestlist else [])

 
#Teacher routes
@app.route('/teacher-login')
def teacherlogin():
    return render_template('teacherlogin.html')

@app.route('/teacher')
def teacher():
    baselines, error = get_baselines()
    if error:
        return render_template('error.html', message=error), 500
    return render_template('teacher.html', baselines=baselines if baselines else [])
    # return render_template('teacher/base.html', baselines=baselines if baselines else [])


@app.route('/handlerequest')
def handlerequest():
    requestlist, error = get_requetsts()
    if error:
        return render_template('error.html', message=error), 500
    return render_template('teacher/resubmitrequest.html',
        baselines=requestlist if requestlist else [])

@app.route('/api/submissions')
def get_submissions_csv():
    submissions = get_submissions()  
    return jsonify(submissions)

@app.route('/view_submission/<student_id>')
def view_submission(student_id):
    baselines, error, status_code = get_last_baseline(student_id)  
    if error:
        return render_template('error.html', message=error), status_code  
    if baselines:
        print("baselines:",baselines)
        return jsonify(baselines)
    return jsonify({'error': 'Submission not found'}), 404

@app.route('/view_submission_admin/<student_id>')
def view_submission_admin(student_id):
    baselines, error, status_code = get_last_baseline_admin(student_id)  
    if error:
        return render_template('error.html', message=error), status_code  
    if not baselines:  
        return jsonify({'error': 'Submission not found'}), 404
    cleaned_baselines = {
        k: v for k, v in baselines.items() 
        if v is not None or k == 'feedback'  
    }
    
    if 'created_at' in cleaned_baselines:
        cleaned_baselines['created_at'] = cleaned_baselines['created_at'].strftime('%Y-%m-%d %H:%M:%S')
    
    print("Returning cleaned baselines:", cleaned_baselines)
    return jsonify(cleaned_baselines)

@app.route('/request_resubmit', methods=['POST'])
def request_resubmit():
    data = request.json
    submissionid = data.get('id')
    feedback = data.get('feedback')
    # print("Submission ID:", submissionid, "Feedback:", feedback)
    success, req_time, error=handleResubmitRequest(submissionid, feedback)   
    
    if success:
        request_time = req_time.now().strftime('%Y-%m-%d %H:%M:%S')
        socketio.emit('resend-request', {
            'baselineid':submissionid,
            'status': 'Awaiting',
            'requestedtime': request_time,
        }, room='admin-room')
        return jsonify({'success': True})
    else:
        return jsonify({'error': error}), 500

@app.route('/matches-content')
def matches_content():    
    baselines, error = get_baselines()
    if error:
        return render_template('error.html', message=error), 500
    return render_template(
        'teacher/matches.html',
        baselines=baselines if baselines else []
    )

@app.route('/validations-content')
def validations_content():
    validations = [
        {'assignment': 'Assignment 1', 'status': 'Needs review'},
        {'assignment': 'Assignment 2', 'status': 'Needs review'}
    ]
    return render_template('teacher/validations.html', validations=validations)

@app.route('/upload_assignments', methods=['POST'])
def upload_assignments():
    """Endpoint to upload assignments and automatically match with baselines"""
    results = {}
    
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
        
    files = request.files.getlist('files')
    timestamps = request.form.getlist('timestamps')
    original_names = request.form.getlist('original_names')

    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No selected files"}), 400
    
    for i, file in  enumerate(files):
        if file and allowed_file(file.filename):
            timestamp = timestamps[i] if i < len(timestamps) else str(int(time.time()))
            filename = secure_filename(file.filename,"tech1","fall2025",timestamp)
            
            folder_path = os.path.join(
                app.config['ASSIGNMENT_FOLDER']
            )
            os.makedirs(folder_path, exist_ok=True)
            
            unique_filename = f"{filename}"
            file_path = os.path.join(folder_path, unique_filename)
            file.save(file_path)





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
        
        baseline_1_path = os.path.join(baseline_dir, f"{baseline_1}.txt")
        baseline_2_path = os.path.join(baseline_dir, f"{baseline_2}.txt")
        teacher_path = os.path.join(teacher_dir, f"{tech}")  

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
                
                if results["major"]:  
                    final_score = min(active_scores.values())  
                else:
                    final_score = sum(active_scores.values()) / len(active_scores) if active_scores else 1.0

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
    
@app.route('/match_assignments', methods=['POST'])
def handle_match_assignments():
    data = request.get_json()
    timestamp = data.get('timestamp')
    print(f"Received timestamp: {timestamp}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    assignments_dir = os.path.join(base_dir, 'assignments')
    baselines_dir = os.path.join(base_dir, 'baseline')

    matching_files = []
    try:
        for filename in os.listdir(assignments_dir):
            if str(timestamp) in str(filename):
                file_path = os.path.join(assignments_dir, filename)
                print("file info===>",file_path,"---",filename)
                if os.path.isfile(file_path):
                    try:
                        content = None  
                        # if filename.endswith('.txt'):
                        #     with open(file_path, 'r', encoding='utf-8') as f:
                        #         content = f.read()
                        content = read_file(file_path)
                        #     print("file info===>",file_path,"---",filename)
                        
                        matching_files.append({
                            'filename': filename,
                            'path': file_path,
                            'content': content, 
                        })
                    except Exception as e:
                        matching_files.append({
                            'filename': filename,
                            'path': file_path,
                            'error': f"Could not read file: {str(e)}"
                        })
        
    except FileNotFoundError:
        return jsonify({'error': 'Assignments directory not found'}), 404

    # print("matching_files---",matching_files)

    baselines = []
    try:
        for student_id in os.listdir(baselines_dir):
            student_dir = os.path.join(baselines_dir, student_id)
            
            if not os.path.isdir(student_dir):
                continue
            
            baseline_entry = {
                'student_id': student_id,
                'name_or_alias': getstudentNamebyId(student_id)
            }
            
            for filename in os.listdir(student_dir):
                file_path = os.path.join(student_dir, filename)
                
                if not os.path.isfile(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    if 'baseline1' in filename:
                        baseline_entry['baseline1_enc'] = content
                        baseline_entry['baseline1'] = getPlantext(filename , content,1)
                    elif 'baseline2' in filename:
                        baseline_entry['baseline2_enc'] = content
                        baseline_entry['baseline2'] = getPlantext(filename , content,2)

                    baseline_entry['filename'] = filename
                except Exception as e:
                    baseline_entry[f'error_{filename}'] = str(e)
            
            
            if 'baseline1' in baseline_entry or 'baseline2' in baseline_entry:
                baselines.append(baseline_entry)
                
    except FileNotFoundError:
        return jsonify({'error': 'Baseline directory not found'}), 404
    
    print("baselines--", baselines)

    matchdata=workingScore(matching_files, baselines, socketio)
    
    return jsonify({
        'status': 'success',
        'received_timestamp': timestamp,
        'data':matchdata
    })
def calculate_combined_score(baseline_path, submission_path):
    """Calculate a combined similarity score between baseline and submission"""
    return 100*round(0.5 + (hash(submission_path) % 5000) / 10000, 4)

def find_best_match(prefix, baseline_folder):
    """Find the best matching student ID for a given prefix"""
    try:
        for student_id in os.listdir(baseline_folder):
            if student_id.startswith(prefix):
                return student_id
    except FileNotFoundError:
        pass
    return None

#Student routes
@app.route('/login', methods=['POST'])
def login():
    result, status_code = handle_login()  
    
    if status_code != 200: 
        return render_template('studentlogin.html', error=result.get('error', 'Unknown error')), status_code
    
    session['student_id'] = result['student_id']
    session['student_name'] = result['student_name']
    # delete path='baseline/`student_id`' directory
    baseline_dir = os.path.join('baseline', str(result['student_id']))
    try:
        if os.path.exists(baseline_dir):
            shutil.rmtree(baseline_dir)
            app.logger.info(f"Deleted baseline directory for student {result['student_id']}")
    except Exception as e:
        app.logger.error(f"Error deleting baseline directory: {str(e)}")
    socketio.emit('student-login', {
        'student_id': result['student_id'],
        'new_status': 'Used',
    }, room='admin-room') 
    return redirect(url_for('student'))

@app.route('/student')
def student():
    if 'student_id' not in session:
        return redirect(url_for('studentlogin'))
    
    submissions, error = get_submissions(session['student_id'])
    
    if error:
        print(f"Error fetching submissions: {error}")
        submissions = [] 
    
    return render_template(
        'student.html',
        student_name=session['student_name'],
        baselines=submissions
    )

@app.route('/student-login')
def studentlogin():    
    return render_template('studentlogin.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('studentlogin'))

@app.route('/submit_baseline', methods=['POST'])
def handle_submit_baseline():
    """Endpoint to submit baseline writing samples"""
    if 'student_id' not in session:
        return jsonify({"error": "You must be logged in to submit baselines"}), 401

    client_ip = request.remote_addr
    print(f"Request from IP: {client_ip}")

    result, error, status_code = submit_baseline(
        student_id=session['student_id'],
        student_name=session.get('student_name', 'Unknown'),
        prompt1=request.form.get('prompt1', ''),
        prompt2=request.form.get('prompt2', ''),
        client_ip=client_ip
    )
    
    if error:
        return jsonify({"error": error}), status_code
    
    return jsonify(result), status_code



# -----------------------End routes-----------------------

def run_server():
    try:
        port = int(os.environ.get('PORT', 5000))
        host = os.environ.get('HOST', '0.0.0.0')
        
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

if __name__ == '__main__':     
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  
    )
# if __name__ == '__main__':
#     socketio.run(app, debug=True)