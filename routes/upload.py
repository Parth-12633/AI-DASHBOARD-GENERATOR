from flask import Blueprint, request, jsonify, send_file, session
import os
import uuid
from werkzeug.utils import secure_filename
from models.database import db, Dataset
from services.data_analyzer_service import process_file
from utils.helpers import allowed_file, clean_json_response

upload_bp = Blueprint('upload', __name__, url_prefix='/api')

@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload CSV or Excel files only.'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join('uploads', unique_filename)
        file.save(file_path)
        
        print(f"File saved to: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        print(f"File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'}")
        
        # Process the file with enhanced loading
        results = process_file(file_path)
        if 'error' in results:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': results['error']}), 400
        
        # Save to database only after successful file processing
        dataset = Dataset(
            user_id=session['user_id'],
            filename=unique_filename,
            original_filename=filename,
            file_size=os.path.getsize(file_path)
        )
        db.session.add(dataset)
        db.session.commit()
        
        # Prepare enhanced response
        response = {
            'success': True,
            'filename': unique_filename,
            'original_filename': filename,
            **results
        }
        
        return clean_json_response(response)
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@upload_bp.route('/download_report/<filename>')
def download_report(filename):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    report_path = f"reports/{filename}"
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True)
    else:
        return jsonify({'error': 'Report not found'}), 404
