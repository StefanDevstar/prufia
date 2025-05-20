import os
import base64
from app.services.admin.common import getTime
from app.services.db.mysql import db_connection
from app.services.security.protect import decrypt, generate_key
from app.services.ai_engine.score import (
    detect_grammar_fixes_only,
    detect_minor_edits,
    detect_structural_changes,
    detect_major_rewrite,
    detect_behavioral_inconsistency
)

def getPlantext(filename, content, id):
    conn = db_connection()
    try:
        with conn.cursor() as cursor:
            if id == 1:  
                cursor.execute("""
                    SELECT salt
                    FROM submissions
                    WHERE baseline_1_path LIKE %s
                    """, (f"%{filename}",))  
            elif id == 2:  
                cursor.execute("""
                    SELECT salt
                    FROM submissions
                    WHERE baseline_2_path LIKE %s
                    """, (f"%{filename}",))
            else:
                return None, "Invalid ID parameter", 400

            results = cursor.fetchone()
            
            if results:
                salt = results[0]
                
                if isinstance(salt, str):
                    try:
                        salt = base64.b64decode(salt)
                    except:
                        salt = salt.encode('latin-1')
                
                if len(salt) == 255:
                    salt = salt[:16]
                if len(salt) != 16:
                    return None, f"Invalid final salt length: {len(salt)} bytes", 400
                
                password = os.getenv("SEC_PROTECT", "default-fallback") 
                key = generate_key(password, salt)
                
                return decrypt(content, key), None, None
                
            return None, "File not found in database", 404

    except Exception as db_error:
        return None, f"Database error: {str(db_error)}", 500
    finally:
        conn.close()


def workingScore(assesses, baselines):
    matchresult = []
    
    for assess in assesses:
        for baseline in baselines:
            try:
                gmarks, rmarks, smarks, mmarks = [], [], [], []
                
                if getattr(baseline, 'baseline1', None) and len(baseline.baseline1) > 0:
                    content1 = baseline.baseline1[0]
                    gmarks.append(detect_grammar_fixes_only(assess.content, content1))
                    rmarks.append(detect_minor_edits(assess.content, content1))
                    smarks.append(detect_structural_changes(assess.content, content1))
                    mmarks.append(detect_major_rewrite(assess.content, content1))
                
                if getattr(baseline, 'baseline2', None) and len(baseline.baseline2) > 0:
                    content2 = baseline.baseline2[0]
                    gmarks.append(detect_grammar_fixes_only(assess.content, content2))
                    rmarks.append(detect_minor_edits(assess.content, content2))
                    smarks.append(detect_structural_changes(assess.content, content2))
                    mmarks.append(detect_major_rewrite(assess.content, content2))
                
                gmark = round(sum(gmarks)/len(gmarks)) if gmarks else 0
                minor = round(sum(rmarks)/len(rmarks)) if rmarks else 0
                structural = round(sum(smarks)/len(smarks)) if smarks else 0
                major = round(sum(mmarks)/len(mmarks)) if mmarks else 0
                
                results = {
                    "major": major,
                    "structural": structural,
                    "minor": minor,
                    "grammar": gmark
                }
                
                weights = {
                    "grammar": 0.95,
                    "minor": 0.85,
                    "structural": 0.62,
                    "major": 0.30
                }
                
                active_scores = {k: weights[k] for k in results if results[k] == 1}
                
                if results["major"] == 1:  
                    final_score = min(active_scores.values()) if active_scores else 0
                else:
                    final_score = (sum(active_scores.values()) / len(active_scores)) if active_scores else 1.0
                
                if final_score >= 0.85:
                    label, flag = "Match", "green"
                elif final_score >= 0.7:
                    label, flag = "Moderate Match", "yellow"
                else:
                    label, flag = "Unclear", "red"
                
                parts = assess['filename'].split('-')
                if len(parts) >= 4:
                    filename, teacher, semester, timestamp_f = parts[0], parts[1], parts[2], parts[3]
                    timestamp = getTime(timestamp_f.split('.')[0])  
                else:
                    raise ValueError(f"Invalid filename format: {assess.filename}")
                
                item = {
                    "flag": flag,
                    "score": int(final_score * 100),
                    "filename": filename,
                    "student_id": baseline['student_id'],
                    "time": timestamp,
                    "teacher": teacher,
                    "semester": semester,
                    "label": label
                }
                matchresult.append(item)  
                
            except (AttributeError, IndexError, ValueError) as e:
                print(f"Error processing {baseline['student_id']}: {str(e)}")
                continue
    
    return matchresult

