from flask import request
from app.services.db.mysql import db_connection

def handle_login():
    """
    Handles student login authentication
    Returns:
        tuple: (dict response, int status_code)
    """
    name = request.form.get('name')
    code = request.form.get('code')

    if not name or not code:
        return {'error': 'Name and code are required'}, 400

    conn = None
    try:
        conn = db_connection()
        with conn.cursor() as cursor:
            # 1. Fetch student
            cursor.execute(
                "SELECT id, name_or_alias FROM students WHERE name_or_alias=%s",
                (name,)
            )
            student = cursor.fetchone()

            if not student:
                return {'error': 'Invalid credentials'}, 401

            student_id = student[0]

            # 2. Check passcode
            cursor.execute(
                "SELECT passcode, used FROM passcode WHERE stdId=%s",
                (student_id,)
            )
            result = cursor.fetchone()

            if not result:
                return {'error': 'Passcode not found'}, 403

            stored_passcode, used = result

            if used:
                return {'error': 'Passcode already used'}, 403
            if code != stored_passcode:
                return {'error': 'Incorrect passcode'}, 401

            # 3. Update passcode - ADD DEBUGGING
            print(f"Attempting to update passcode for student {student_id}")
            cursor.execute(
                "UPDATE passcode SET used=1 WHERE stdId=%s AND used=0",
                (student_id,)
            )
            affected_rows = cursor.rowcount

            if affected_rows == 0:
                print("WARNING: No rows updated - possible race condition")
                return {'error': 'Passcode already used'}, 403

            # 4. EXPLICITLY COMMIT THE TRANSACTION
            conn.commit()

            return {
                'student_id': student_id,
                'student_name': student[1]
            }, 200

    except Exception as e:
        print(f"Login error: {e}")
        if conn:
            conn.rollback()
        return {'error': 'Login failed'}, 500

    finally:
        if conn:
            conn.close()