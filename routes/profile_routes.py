# routes/profile_routes.py

import logging
import os
import uuid
from flask import Blueprint, request, jsonify,current_app
import psycopg2
from services.db_service import get_db
from utils.file_utils import allowed_file, save_audio_file
import datetime
# from flask import current_app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
profile_routes = Blueprint('profile_routes', __name__)

# Configuration

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'webm'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size


# current_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# current_app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


@profile_routes.route('/add-profile', methods=['POST'])
def add_profile_route():
    try:
        print(current_app.root_path)
        UPLOAD_FOLDER = os.path.join(current_app.root_path, 'static', 'uploads')
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        if 'voiceName' not in request.form or not request.form['voiceName']:
            return jsonify({"error": "Voice Name is required"}), 400
        if 'language' not in request.form or not request.form['language']:
            return jsonify({"error": "Language is required"}), 400
        if 'audio' not in request.files:
            return jsonify({"error": "Audio file is required"}), 400

        language = request.form['language']
        voice_name = request.form['voiceName']
        audio_file = request.files['audio']
        # Use created_at from form if provided, else default to now
        created_at_str = request.form.get('created_at') 

        if audio_file.filename == '':
            return jsonify({"error": "No selected audio file"}), 400

        if not allowed_file(audio_file.filename):
            return jsonify({"error": "Invalid audio file type. Allowed types: mp3, wav, webm"}), 400

        original_filename = audio_file.filename
        name_part, extension = os.path.splitext(original_filename)
        safe_voice_name = voice_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        unique_id = uuid.uuid4().hex
        filename = f"{safe_voice_name}_{unique_id}{extension}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        audio_file.save(file_path)
        audio_path_in_db = os.path.join('uploads', filename)

        created_at_timestamp = None
        if created_at_str:
            try:
                created_at_timestamp = datetime.datetime.strptime(created_at_str, '%Y-%m-%d')
            except ValueError:
                 # If format is YYYY-MM-DDTHH:MM:SS or similar from a datetime-local, try parsing that
                try:
                    created_at_timestamp = datetime.datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                except ValueError:
                    return jsonify({"error": "Invalid date format for created_at. Use YYYY-MM-DD or ISO format."}), 400
        else:
            created_at_timestamp = datetime.datetime.now()

        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO voice_profiles (language, voice_name, audio_path, created_at)
                    VALUES (%s, %s, %s, %s)
                """, (language, voice_name, audio_path_in_db, created_at_timestamp))
                conn.commit()

        return jsonify({"status": "success", "message": "Profile saved successfully"}), 201

    except Exception as e:
        logger.error(f"Error in /add-profile: {str(e)}")
        return jsonify({"error": f"Server error in /add-profile: {str(e)}"}), 500

@profile_routes.route('/api/profiles', methods=['GET'])
def get_profiles():
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, language, voice_name, audio_path, created_at FROM voice_profiles ORDER BY created_at DESC LIMIT 5")
                profiles_raw = cur.fetchall()
                column_names = [desc[0] for desc in cur.description]
                profiles = [dict(zip(column_names, row)) for row in profiles_raw]
                
                for profile in profiles:
                    if isinstance(profile['created_at'], datetime.datetime):
                        profile['created_at'] = profile['created_at'].isoformat() # Use ISO format for consistency
                
                return jsonify(profiles), 200
    except Exception as e:
        logger.error(f"Error in /api/profiles: {str(e)}")
        return jsonify({"error": "Failed to fetch profiles"}), 500

@profile_routes.route('/api/profiles/<int:profile_id>', methods=['GET'])
def get_profile(profile_id):
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, language, voice_name, audio_path, created_at FROM voice_profiles WHERE id = %s", (profile_id,))
                profile_raw = cur.fetchone()
                if profile_raw is None:
                    return jsonify({"error": "Profile not found"}), 404
                
                column_names = [desc[0] for desc in cur.description]
                profile = dict(zip(column_names, profile_raw))
                
                if isinstance(profile['created_at'], datetime.datetime):
                    profile['created_at'] = profile['created_at'].isoformat()
                
                return jsonify(profile), 200
    except Exception as e:
        logger.error(f"Error fetching profile {profile_id}: {str(e)}")
        return jsonify({"error": "Failed to fetch profile"}), 500

@profile_routes.route('/api/profiles/<int:profile_id>/edit', methods=['POST'])
def update_profile(profile_id):
    UPLOAD_FOLDER = os.path.join(current_app.root_path, 'static', 'uploads')
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT audio_path, created_at FROM voice_profiles WHERE id = %s", (profile_id,))
                current_profile_data = cur.fetchone()
                if current_profile_data is None:
                    return jsonify({"error": "Profile not found"}), 404
                
                current_audio_path_db = current_profile_data[0]
                current_created_at_db = current_profile_data[1]

                voice_name = request.form.get('voiceName')
                language = request.form.get('language')
                # Frontend sends 'created_at' for the last modified date field
                created_at_str = request.form.get('created_at') 
                new_audio_file = request.files.get('audio')

                update_fields = {}
                if voice_name:
                    update_fields['voice_name'] = voice_name
                if language:
                    update_fields['language'] = language
                
                update_fields['audio_path'] = current_audio_path_db # Default to current

                if new_audio_file and new_audio_file.filename != '':
                    if not allowed_file(new_audio_file.filename):
                        return jsonify({"error": "Invalid audio file type for update. Allowed types: mp3, wav, webm"}), 400
                    
                    if current_audio_path_db:
                        old_file_full_path = os.path.join(UPLOAD_FOLDER, os.path.basename(current_audio_path_db))
                        if os.path.exists(old_file_full_path):
                            try: os.remove(old_file_full_path)
                            except Exception as e_del: logger.error(f"Error deleting old audio file {old_file_full_path}: {str(e_del)}")

                    original_filename = new_audio_file.filename
                    name_part, extension = os.path.splitext(original_filename)
                    safe_voice_name_for_file = (voice_name or update_fields.get('voice_name', 'profile')).replace(" ", "_").replace("/", "_").replace("\\", "_")
                    unique_id = uuid.uuid4().hex
                    filename = f"{safe_voice_name_for_file}_{unique_id}{extension}"
                    file_path = os.path.join(UPLOAD_FOLDER, filename)
                    new_audio_file.save(file_path)
                    update_fields['audio_path'] = os.path.join('uploads', filename)

                if created_at_str:
                    try:
                        update_fields['created_at'] = datetime.datetime.strptime(created_at_str, '%Y-%m-%d')
                    except ValueError:
                        try:
                            update_fields['created_at'] = datetime.datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                        except ValueError:
                            return jsonify({"error": "Invalid date format for Last Modified Date. Use YYYY-MM-DD or ISO format."}), 400
                else:
                    update_fields['created_at'] = current_created_at_db

                if not update_fields and not new_audio_file: # Check if any actual changes were made
                     return jsonify({"status": "no_change", "message": "No fields to update"}), 200

                set_clause_parts = []
                values_list = []
                for key, value in update_fields.items():
                    set_clause_parts.append(f"{key} = %s")
                    values_list.append(value)
                
                if not set_clause_parts: # If only audio was changed, but no other fields
                    if new_audio_file and new_audio_file.filename != '': # Ensure audio was actually changed
                        set_clause_parts.append(f"audio_path = %s")
                        values_list.append(update_fields['audio_path'])
                    else:
                        return jsonify({"status": "no_change", "message": "No fields to update"}), 200

                set_clause = ", ".join(set_clause_parts)
                values_list.append(profile_id)
                
                cur.execute(f"UPDATE voice_profiles SET {set_clause} WHERE id = %s", tuple(values_list))
                conn.commit()
                return jsonify({"status": "success", "message": "Profile updated successfully"}), 200
    except Exception as e:
        logger.error(f"Error updating profile {profile_id}: {str(e)}")
        return jsonify({"error": f"Server error updating profile: {str(e)}"}), 500

# Corrected DELETE route to match frontend call
@profile_routes.route('/api/profiles/<int:profile_id>', methods=['DELETE'])
def delete_profile(profile_id):
    try:
        UPLOAD_FOLDER = os.path.join(current_app.root_path, 'static', 'uploads')
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT audio_path FROM voice_profiles WHERE id = %s", (profile_id,))
                profile_to_delete = cur.fetchone()

                if profile_to_delete is None:
                    return jsonify({"error": "Profile not found to delete"}), 404
                
                audio_path_to_delete_db = profile_to_delete[0]

                cur.execute("DELETE FROM voice_profiles WHERE id = %s", (profile_id,))
                conn.commit()

                if cur.rowcount == 0: # Check if delete actually happened
                    logger.warning(f"Profile ID {profile_id} was not found in DB for deletion, though it was fetched.")
                    # This case should ideally not happen if fetch was successful, but good to log.
                    return jsonify({"error": "Profile could not be deleted from database, might have been deleted already."}), 404

                if audio_path_to_delete_db:
                    file_to_delete_full_path = os.path.join(UPLOAD_FOLDER, os.path.basename(audio_path_to_delete_db))
                    if os.path.exists(file_to_delete_full_path):
                        try:
                            os.remove(file_to_delete_full_path)
                        except Exception as e_del_file:
                            logger.error(f"Error deleting audio file {file_to_delete_full_path} for profile {profile_id}: {str(e_del_file)}")
                            # Log error but consider DB deletion successful if rowcount was > 0
                            return jsonify({"status": "success_db_only", "message": f"Profile deleted from DB, but failed to delete audio file: {str(e_del_file)}"}), 200 # Partial success
                    else:
                        logger.warning(f"Audio file {file_to_delete_full_path} not found for deleted profile {profile_id}")
                
                return jsonify({"status": "success", "message": "Profile deleted successfully"}), 200
    except psycopg2.Error as db_err:
        logger.error(f"Database error deleting profile {profile_id}: {str(db_err)}")
        return jsonify({"error": f"Database error: {str(db_err)}"}), 500
    except Exception as e:
        logger.error(f"General error deleting profile {profile_id}: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# This /save-profile route is likely redundant if /add-profile is used by the main form/modal.
# It's kept here for now as it was in the original provided code.
@profile_routes.route('/save-profile', methods=['POST'])
def save_profile(): 
    try:
        UPLOAD_FOLDER = os.path.join(current_app.root_path, 'static', 'uploads')
        language = request.form['language']
        voice_name = request.form['voiceName']
        audio_file = request.files['audio']
        created_at_str = request.form.get('created_at') # Assuming it might be passed
        
        if not language or not voice_name or not audio_file:
            return jsonify({"error": "Missing required fields"}), 400

        if not allowed_file(audio_file.filename):
            return jsonify({"error": "Invalid audio file type"}), 400

        original_filename = audio_file.filename
        name_part, extension = os.path.splitext(original_filename)
        safe_voice_name = voice_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        unique_id = uuid.uuid4().hex
        filename = f"{safe_voice_name}_{unique_id}{extension}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        audio_file.save(file_path)
        audio_path_in_db = os.path.join('uploads', filename)

        created_at_timestamp = None
        if created_at_str:
            try: created_at_timestamp = datetime.datetime.strptime(created_at_str, '%Y-%m-%d')
            except ValueError: 
                try: created_at_timestamp = datetime.datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                except ValueError: return jsonify({"error": "Invalid date format for created_at. Use YYYY-MM-DD or ISO format."}), 400
        else: created_at_timestamp = datetime.datetime.now()

        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO voice_profiles (language, voice_name, audio_path, created_at)
                    VALUES (%s, %s, %s, %s)
                """, (language, voice_name, audio_path_in_db, created_at_timestamp))
                conn.commit()

        return jsonify({"status": "success", "message": "Profile saved successfully via /save-profile"}), 200
    except Exception as e:
        logger.error(f"Error in /save-profile: {str(e)}")
        return jsonify({"error": str(e)}), 500

