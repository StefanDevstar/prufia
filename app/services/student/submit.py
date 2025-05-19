import os
import uuid
from flask import current_app
from app.services.db.mysql import db_connection
from app.services.security.protect import encrypt, generate_key

def submit_baseline(student_id, student_name, prompt1, prompt2, client_ip, semester_id="2025-Pilot"):
    try:
        if not prompt1 or not prompt2:
            return None, "Both prompts are required", 400
        
        # Generate new salt for each submission
        salt = os.urandom(16)
        key = generate_key(os.getenv("SEC_PROTECT", "default-fallback"), salt)
        
        # Encrypt prompts
        encrypted_p1 = encrypt(prompt1, key)
        encrypted_p2 = encrypt(prompt2, key)

        # Prepare storage
        folder = os.path.join(current_app.root_path, 
                           current_app.config['BASELINE_FOLDER'], 
                           str(student_id))
        os.makedirs(folder, exist_ok=True)
        
        # Save encrypted files
        file_id = str(uuid.uuid4())
        p1_path = os.path.join(folder, f"baseline1_{file_id}.enc")
        p2_path = os.path.join(folder, f"baseline2_{file_id}.enc")
        
        with open(p1_path, 'w') as f:
            f.write(encrypted_p1)
        with open(p2_path, 'w') as f:
            f.write(encrypted_p2)

        # Store in database
        conn = db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO submissions 
                    (student_id, baseline_1_path, baseline_2_path, semester_id, salt,ip) 
                    VALUES (%s, %s, %s, %s, %s, %s)""",
                    (student_id, p1_path, p2_path, semester_id, salt,client_ip)
                )
                conn.commit()
                
            return {
                "status": "success",
                "message": "Encrypted baselines saved successfully",
                "paths": {
                    "baseline1": p1_path,
                    "baseline2": p2_path
                }
            }, None, 200
            
        except Exception as db_error:
            conn.rollback()
            # Clean up files if DB operation failed
            for path in [p1_path, p2_path]:
                if os.path.exists(path):
                    os.remove(path)
            return None, f"Database error: {str(db_error)}", 500
        finally:
            conn.close()
            
    except Exception as e:
        return None, f"Submission error: {str(e)}", 500
    

def getstudents():
    conn = db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM students")
            result = cursor.fetchall()  # Fetch all records
            
        return {
            "status": "success",
            "message": "Get students successfully",
            "data": result
        }, None, 200

    except Exception as db_error:
        conn.rollback()
        return None, f"Database error: {str(db_error)}", 500
    finally:
        conn.close()