import pymysql
import os
import base64
from app.services.teacher.business import getPlantext
from app.services.db.mysql import db_connection
from app.services.security.protect import decrypt, generate_key

def get_baselines():
    """
    Retrieves all student submissions with related data
    Returns:
        list: List of submission records
        str: Error message if any
    """
    conn = None
    try:
        conn = db_connection()
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    s.id, s.student_id, st.name_or_alias,
                    s.baseline_1_path, s.baseline_2_path,
                    s.created_at, s.submission_path,
                    s.score_baseline_1, s.score_baseline_2,
                    s.final_score, s.trust_flag,
                    s.interpretation
                FROM submissions s
                JOIN students st ON s.student_id = st.id
                ORDER BY s.created_at DESC
            """)
            return cursor.fetchall(), None
            
    except pymysql.MySQLError as e:
        print(f"Database error: {str(e)}")
        return None, str(e)
        
    finally:
        if conn:
            conn.close()

def get_submissions(student_id):
    """
    Retrieves all submissions for a specific student
    Args:
        student_id: ID of the student
    Returns:
        tuple: (list of submissions, error message)
    """
    conn = None
    try:
        conn = db_connection()
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM submissions WHERE student_id=%s ORDER BY created_at DESC",
                (student_id,)
            )
            return cursor.fetchall(), None
    except pymysql.MySQLError as e:
        print(f"Database error: {str(e)}")
        return None, str(e)
    finally:
        if conn:
            conn.close()

def get_last_baseline(student_id):
    """
    Retrieves a submission for a specific student and decrypts the baseline files
    Args:
        student_id: ID of the student
    Returns:
        tuple: (dict containing decrypted baselines, error message, status_code)
    """
    conn = None
    try:
        conn = db_connection()
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM submissions WHERE student_id=%s ORDER BY created_at DESC LIMIT 1",
                (student_id,)
            )
            submission = cursor.fetchone()
            
            if not submission:
                return None, "No submission found for this student", 404
            
            baseline1 = None
            baseline2 = None
            errors = []
            
            # Process salt
            salt = submission.get('salt')
            if salt:
                if isinstance(salt, str):
                    try:
                        salt = base64.b64decode(salt)
                    except:
                        salt = salt.encode('latin-1')
                
                if len(salt) == 255:
                    salt = salt[:16]
                if len(salt) != 16:
                    return None, f"Invalid salt length: {len(salt)} bytes", 400
            
            # Decrypt baseline2 if exists
            if submission.get('baseline_2_path') and salt:
                try:
                    with open(submission['baseline_2_path'], 'r', encoding='utf-8') as f:
                        encrypted_data = f.read()
                        filename = os.path.basename(submission['baseline_2_path'])
                        decrypted_data, error, status = getPlantext(filename, encrypted_data, 2)
                        if decrypted_data:
                            baseline2 = decrypted_data
                        else:
                            errors.append(f"baseline2: {error}")
                except Exception as e:
                    errors.append(f"baseline2: {str(e)}")
            
            # Decrypt baseline1 if exists
            if submission.get('baseline_1_path') and salt:
                try:
                    with open(submission['baseline_1_path'], 'r', encoding='utf-8') as f:
                        encrypted_data = f.read()
                        filename = os.path.basename(submission['baseline_1_path'])
                        decrypted_data, error, status = getPlantext(filename, encrypted_data, 1)
                        if decrypted_data:
                            baseline1 = decrypted_data
                        else:
                            errors.append(f"baseline1: {error}")
                except Exception as e:
                    errors.append(f"baseline1: {str(e)}")           

            
            result = {
                "baseline1": baseline1,
                "baseline2": baseline2,
                # "submission_id": submission.get('id'),
                # "student_id": submission.get('student_id'),
                "created_at": submission.get('created_at')
            }
            
            error_msg = ", ".join(errors) if errors else None
            return result, error_msg, (400 if errors else 200)
            
    except pymysql.MySQLError as e:
        return None, f"Database error: {str(e)}", 500
    except Exception as e:
        return None, f"Processing error: {str(e)}", 500
    finally:
        if conn:
            conn.close()



def get_last_baseline_admin(student_id):
    """
    Retrieves a submission for a specific student and decrypts the baseline files
    Args:
        student_id: ID of the student
    Returns:
        tuple: (dict containing decrypted baselines, error message, status_code)
    """
    conn = None
    try:
        conn = db_connection()
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    s.*,
                    rr.feedback,
                    rr.requested_time,
                    rr.id AS resubmit_id
                FROM 
                    submissions s
                LEFT JOIN 
                    resubmit_request rr ON s.id = rr.base_id
                WHERE 
                    s.student_id = %s 
                ORDER BY 
                    s.created_at DESC 
                LIMIT 1
            """, (student_id,))
            submission = cursor.fetchone()
            
            if not submission:
                return None, "No submission found for this student", 404
            
            baseline1 = None
            baseline2 = None
            errors = []
            
            # Process salt
            salt = submission.get('salt')
            if salt:
                if isinstance(salt, str):
                    try:
                        salt = base64.b64decode(salt)
                    except:
                        salt = salt.encode('latin-1')
                
                if len(salt) == 255:
                    salt = salt[:16]
                if len(salt) != 16:
                    return None, f"Invalid salt length: {len(salt)} bytes", 400
            
            # Decrypt baseline2 if exists
            if submission.get('baseline_2_path') and salt:
                try:
                    with open(submission['baseline_2_path'], 'r', encoding='utf-8') as f:
                        encrypted_data = f.read()
                        filename = os.path.basename(submission['baseline_2_path'])
                        decrypted_data, error, status = getPlantext(filename, encrypted_data, 2)
                        if decrypted_data:
                            baseline2 = decrypted_data
                        else:
                            errors.append(f"baseline2: {error}")
                except Exception as e:
                    errors.append(f"baseline2: {str(e)}")
            
            # Decrypt baseline1 if exists
            if submission.get('baseline_1_path') and salt:
                try:
                    with open(submission['baseline_1_path'], 'r', encoding='utf-8') as f:
                        encrypted_data = f.read()
                        filename = os.path.basename(submission['baseline_1_path'])
                        decrypted_data, error, status = getPlantext(filename, encrypted_data, 1)
                        if decrypted_data:
                            baseline1 = decrypted_data
                        else:
                            errors.append(f"baseline1: {error}")
                except Exception as e:
                    errors.append(f"baseline1: {str(e)}")           

            
            result = {
                "baseline1": baseline1,
                "baseline2": baseline2,
                "feedback": submission.get('feedback'),
                "requested_time": submission.get('requested_time'),
                "created_at": submission.get('created_at')
            }
            
            error_msg = ", ".join(errors) if errors else None
            return result, error_msg, (400 if errors else 200)
            
    except pymysql.MySQLError as e:
        return None, f"Database error: {str(e)}", 500
    except Exception as e:
        return None, f"Processing error: {str(e)}", 500
    finally:
        if conn:
            conn.close()


def get_requetsts():
    """
    Retrieves resubmission requests with related student and submission info
    Returns:
        list: List of dictionaries containing the requested fields
        str: Error message if any
    """
    conn = None
    try:
        conn = db_connection()
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    s.id AS id,
                    s.student_id,
                    st.name_or_alias,
                    s.created_at AS created_at,
                    rr.status AS status,
                    rr.feedback,
                    rr.created_at AS requested_time,
                    s.baseline_1_path,
                    s.baseline_2_path,
                    s.salt,
                    s.ip,
                    s.semester_id,
                    s.submission_path
                FROM 
                    submissions s
                JOIN 
                    students st ON s.student_id = st.id
                LEFT JOIN 
                    resubmit_request rr ON s.id = rr.base_id
                ORDER BY 
                    s.created_at DESC
            """)
            return cursor.fetchall(), None
    except pymysql.MySQLError as e:
        print(f"Database error: {str(e)}")
        return None, f"Database error: {str(e)}"
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None, f"Unexpected error: {str(e)}"
    finally:
        if conn:
            conn.close()