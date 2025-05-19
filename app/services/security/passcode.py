import string
import random
from flask import request, jsonify
from app.services.db.mysql import db_connection

# def gencode(student_id):
#     try:
#         length = int(request.args.get('length', 9))
#         if length < 1:
#             raise ValueError("Passcode length must be positive")
        
#         chars = string.ascii_letters + string.digits
#         passcode = ''.join(random.choices(chars, k=length))
#         conn = db_connection()
#         try:
#             with conn.cursor() as cursor:
#                 # First check if student already has an active passcode
#                 cursor.execute(
#                     """SELECT id FROM passcode 
#                     WHERE stdId = %s 
#                     AND created_at > NOW() - INTERVAL 24 HOUR""",
#                     (student_id,)
#                 )
#                 if cursor.fetchone():
#                     return jsonify({
#                         'status': 'error',
#                         'message': 'Student already has an active passcode'
#                     }), 400
                
#                 # Insert new passcode
#                 cursor.execute(
#                     """INSERT INTO passcode 
#                     (stdId, passcode, used) 
#                     VALUES (%s, %s, %s)""",
#                     (student_id, passcode, 0)
#                 )
                
#                 # Get the created_at timestamp
#                 cursor.execute(
#                     """SELECT created_at FROM passcode 
#                     WHERE id = LAST_INSERT_ID()"""
#                 )
#                 created_at = cursor.fetchone()[0]
                
#                 conn.commit()
                
#                 return jsonify({
#                     'status': 'success',
#                     'passcode': passcode,
#                     'length': length,
#                     'student_id': student_id,
#                     'created_at': created_at.isoformat(),
#                     'message': f'{length}-character passcode generated successfully'
#                 }), 200
            
#         except Exception as db_error:
#             conn.rollback()
#             return jsonify({
#                 'status': 'error',
#                 'message': f"Database error: {str(db_error)}"
#             }), 500
#         finally:
#             conn.close()        

#     except Exception as e:
#         return jsonify({
#             'status': 'error',
#             'message': str(e)
#         }), 500
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
                    WHERE stdId = %s 
                    AND used = 0""",  # Only consider unused passcodes
                    (student_id,)
                )
                existing_passcode = cursor.fetchone()
                
                if not existing_passcode:
                    # No active passcode - insert new
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
                else:
                    # Active passcode exists - update it
                    cursor.execute(
                        """UPDATE passcode 
                        SET passcode = %s, 
                            created_at = NOW(),
                            used = 0
                        WHERE stdId = %s""",
                        (passcode, student_id)
                    )
                    # Get the created_at timestamp
                    cursor.execute(
                        """SELECT created_at FROM passcode 
                        WHERE stdId = %s""",(student_id)
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
        

def getpasscode():
    conn = db_connection()
    try:
        with conn.cursor() as cursor:  # Removed dictionary=True
            cursor.execute("""
                SELECT 
                    student.id,
                    student.name_or_alias,
                    student.email,
                    passcode.passcode,
                    passcode.used
                FROM students AS student
                JOIN passcode ON student.id = passcode.stdId
            """)
            results = cursor.fetchall()
            
            # Manually convert to dictionaries
            passcodes_list = []
            for row in results:
                passcodes_list.append({
                    'student_id': row[0],
                    'name_or_alias': row[1],
                    'email': row[2],
                    'passcode': row[3],
                    'used': row[4]
                })
            
            return {
                'status': 'success', 
                'data': passcodes_list,
                'count': len(passcodes_list)
            }, None, 200

    except Exception as db_error:
        return None, f"Database error: {str(db_error)}", 500
    finally:
        conn.close()

def handle_reset_all_passcode(student_id=None):
    """
    Reset passcode usage status
    Args:
        student_id: Specific student ID or None for all passcodes
    Returns:
        tuple: (response_dict, status_code)
    """
    conn = None
    try:
        conn = db_connection()
        with conn.cursor() as cursor:
            if student_id:
                cursor.execute(
                    "UPDATE passcode SET used=0 WHERE stdId=%s",
                    (student_id,)
                )
                message = f"Passcode reset for student {student_id}"
            else:
                # Reset all passcodes
                cursor.execute("UPDATE passcode SET used=0")
                message = "All passcodes reset"
            
            affected_rows = cursor.rowcount
            conn.commit()
            
            if affected_rows == 0:
                return {"status": "error", "message": "No passcodes found to reset"}, 404
            
            return {
                "status": "success",
                "message": message,
                "affected_rows": affected_rows
            }, 200
    
    except Exception as e:
        print(f"Error resetting passcode: {e}")
        if conn:
            conn.rollback()
        return {
            "status": "error",
            "message": "Failed to reset passcode",
            "error": str(e)
        }, 500
    
    finally:
        if conn:
            conn.close()