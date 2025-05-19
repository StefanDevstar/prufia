from app.services.db.mysql import db_connection
import pymysql

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