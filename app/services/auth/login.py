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
            # Get student info
            cursor.execute(
                "SELECT id, name_or_alias FROM students WHERE name_or_alias=%s",
                (name,)
            )
            student = cursor.fetchone()

            if not student:
                cursor.execute(
                    "SELECT stdId FROM passcode WHERE passcode=%s",
                    (code,)
                )
                result = cursor.fetchone()
                if result and result[0] == -1:
                    cursor.execute(
                        "INSERT INTO students (name_or_alias) VALUES (%s)",
                        (name, )
                    )
                    # Commit the transaction
                    conn.commit()
                    
                    # Retrieve the last inserted id
                    cursor.execute("SELECT LAST_INSERT_ID()")
                    result = cursor.fetchone()

                    if result:
                        stdId = result[0]  # Fetching the first element in the tuple
                        print(f"Student '{name}' inserted with ID: {stdId}")
                        
                        cursor.execute(
                            "UPDATE passcode SET stdId=%s, used=1 WHERE passcode=%s",
                            (stdId, code,)
                        )
                        affected_rows = cursor.rowcount

                        if affected_rows == 0:
                            print("WARNING: No rows updated - possible race condition")

                            return {'error': 'Passcode already used'}, 403

                        conn.commit()

                        return {
                            'student_id': stdId,
                            'student_name': name
                        }, 200

                    else:
                        print("Failed to retrieve the last inserted ID.")
                        return None
                    
                else:
                    return {'error': 'Invalid credentials'}, 401


            student_id = student[0]

            # Check passcode
            cursor.execute(
                "SELECT passcode, used FROM passcode WHERE stdId=%s",
                (student_id,)
            )
            result = cursor.fetchone()

            if not result:
                return {'error': 'Passcode not found'}, 403

            stored_passcode, used = result

            if code != stored_passcode:
                return {'error': 'Incorrect passcode'}, 401
                
            if used:
                return {'error': 'Passcode already used'}, 403
            

            # Check submission status if exists
            cursor.execute(
                "SELECT id FROM submissions WHERE student_id=%s ORDER BY created_at DESC LIMIT 1",
                (student_id,)
            )
            submission = cursor.fetchone()

            if submission:
                submission_id = submission[0]
                cursor.execute(
                    "SELECT status FROM resubmit_request WHERE base_id=%s",
                    (submission_id,)
                )
                status_result = cursor.fetchone()
                
                if status_result and status_result[0] == 1:  # If status is 1 (pending)
                    return {'error': 'Your submission is pending approval'}, 403

            # Mark passcode as used (runs for both cases - with or without submission)
            cursor.execute(
                "UPDATE passcode SET used=1 WHERE stdId=%s AND used=0",
                (student_id,)
            )
            affected_rows = cursor.rowcount

            if affected_rows == 0:
                print("WARNING: No rows updated - possible race condition")
                return {'error': 'Passcode already used'}, 403

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