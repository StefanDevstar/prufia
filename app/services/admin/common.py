from datetime import datetime
from app.services.db.mysql import db_connection

def getTime(timestamp):
    timestamp_ms = int(timestamp)
    timestamp_sec = timestamp_ms / 1000
    dt = datetime.fromtimestamp(timestamp_sec)
    formatted_date = dt.strftime('%B %d, %Y %H:%M:%S')
    return formatted_date

def updateApprve(baselineid):
    conn = None
    try:
        conn = db_connection()
        with conn.cursor() as cursor:
            # Update the status from 1 (requested) to 2 (approved)
            print("baselineid==",baselineid)
            cursor.execute("""
                UPDATE resubmit_request 
                SET status = 2,
                    approved_at = NOW()  # Add timestamp of approval
                WHERE base_id = %s AND status = 1
                """, (baselineid,))
            
            # Check if the update affected any rows
            updated_row = cursor.fetchone()
            conn.commit()
            
            if updated_row:
                return True, None  # Success
            else:
                return False, "No matching request found or already processed"
                
    except Exception as e:
        if conn:
            conn.rollback()
        return False, str(e)
    finally:
        if conn:
            conn.close()