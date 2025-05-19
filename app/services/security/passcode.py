import string
import random
from flask import request, jsonify
from app.services.db.mysql import db_connection

def gencode(student_id):
    try:
        length = int(request.args.get('length', 9))
        if length < 1:
            raise ValueError("Passcode length must be positive")
        
        chars = string.ascii_letters + string.digits
        passcode = ''.join(random.choices(chars, k=length))
        conn = db_connection()
        try:
            with conn.cursor() as cursor:
                # First check if student already has an active passcode
                cursor.execute(
                    """SELECT id FROM passcode 
                    WHERE stdId = %s AND used = 0 
                    AND created_at > NOW() - INTERVAL 24 HOUR""",
                    (student_id,)
                )
                if cursor.fetchone():
                    return jsonify({
                        'status': 'error',
                        'message': 'Student already has an active passcode'
                    }), 400
                
                # Insert new passcode
                cursor.execute(
                    """INSERT INTO passcode 
                    (stdId, passcode, used) 
                    VALUES (%s, %s, %s)""",
                    (student_id, passcode, 0)
                )
                
                # Get the created_at timestamp
                cursor.execute(
                    """SELECT created_at FROM passcode 
                    WHERE id = LAST_INSERT_ID()"""
                )
                created_at = cursor.fetchone()[0]
                
                conn.commit()
                
                return jsonify({
                    'status': 'success',
                    'passcode': passcode,
                    'length': length,
                    'student_id': student_id,
                    'created_at': created_at.isoformat(),
                    'message': f'{length}-character passcode generated successfully'
                }), 200
            
        except Exception as db_error:
            conn.rollback()
            return jsonify({
                'status': 'error',
                'message': f"Database error: {str(db_error)}"
            }), 500
        finally:
            conn.close()        

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500