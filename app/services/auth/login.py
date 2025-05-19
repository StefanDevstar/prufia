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

    # Input validation
    if not name or not code:
        return {'error': 'Name and code are required'}, 400

    conn = None
    try:
        conn = db_connection()
        with conn.cursor() as cursor:
            # Fetch student by name
            cursor.execute(
                "SELECT id, name_or_alias FROM students WHERE name_or_alias=%s",
                (name,)  # tuple with comma
            )
            student = cursor.fetchone()

            if not student:
                return {'error': 'Invalid credentials'}, 401

            student_id = student[0]

            # Check for an unused passcode for this student
            cursor.execute(
                "SELECT passcode, used FROM passcode WHERE stdId=%s",
                (student_id,)
            )
            result = cursor.fetchone()

            if not result:
                return {'error': 'Your passcode has expired or does not exist.'}, 403

            stored_passcode, used = result

            # Verify passcode
            if used:
                return {'error': 'Passcode has already been used.'}, 403

            if code != stored_passcode:
                return {'error': 'Incorrect passcode.'}, 401

            # Mark passcode as used
            cursor.execute(
                "UPDATE passcode SET used=1 WHERE passcode=%s",
                (stored_passcode,)
            )

            # Return success response
            return {
                'student_id': student[0],
                'student_name': student[1]
            }, 200

    except Exception as e:
        print(f"Login error: {e}")
        return {'error': 'An error occurred during login'}, 500

    finally:
        if conn:
            conn.close()