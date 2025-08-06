from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import json
import os
import uuid
from datetime import datetime
from enhanced_data_analyzer import EnhancedDataAnalyzer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dashboard.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('reports', exist_ok=True)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    file_size = db.Column(db.Integer, nullable=False)

# Routes
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and bcrypt.check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials'})
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Username already exists'})
        
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': 'Email already registered'})
        
        # Create new user
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password_hash=password_hash)
        
        try:
            db.session.add(user)
            db.session.commit()
            return jsonify({'success': True, 'message': 'Registration successful'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': 'Registration failed'})
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
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
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        print(f"File saved to: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        print(f"File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'}")
        
        # Process the file with enhanced loading
        df = load_dataframe_enhanced(file_path)
        if df is None:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': 'Could not read the file. Please ensure it\'s a valid CSV or Excel file.'}), 400
        
        print(f"DataFrame loaded successfully. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Save to database only after successful file processing
        dataset = Dataset(
            user_id=session['user_id'],
            filename=unique_filename,
            original_filename=filename,
            file_size=os.path.getsize(file_path)
        )
        db.session.add(dataset)
        db.session.commit()
        
        # Initialize enhanced analyzer
        analyzer = EnhancedDataAnalyzer(df)
        
        print("Step 1: Analyzer initialized")
        # Clean data
        cleaned_df, cleaning_report = analyzer.clean_data()
        print("Step 2: Data cleaned")
        # Generate enhanced EDA report
        eda_report = generate_enhanced_eda_report(analyzer)
        print("Step 3: EDA report generated")
        # Generate smart visualizations
        charts = analyzer.generate_smart_visualizations()
        print("Step 4: Charts generated")
        # Generate enhanced insights
        insights = generate_enhanced_insights(analyzer)
        print("Step 5: Insights generated")
        # Generate enhanced KPIs
        kpis = generate_enhanced_kpis(analyzer)
        print("Step 6: KPIs generated")
        # Generate PDF report (you can enhance this too)
        pdf_filename = f"{unique_filename.split('.')[0]}_dashboard_report.pdf"
        pdf_path = analyzer.export_dashboard_and_eda_to_pdf(charts, eda_report, filename=pdf_filename)
        print("Step 7: PDF generated")
        
        # Prepare enhanced response
        response = {
            'success': True,
            'filename': unique_filename,
            'original_filename': filename,
            'basic_stats': {
                'original_rows': len(df),
                'columns': len(df.columns),
                'numeric_columns': len(analyzer.numeric_columns),
                'categorical_columns': len(analyzer.categorical_columns),
                'datetime_columns': len(analyzer.datetime_columns)
            },
            'cleaning_report': cleaning_report,
            'eda_report': eda_report,
            'charts': charts,
            'insights': insights,
            'kpis': kpis,
            'pdf_report': pdf_filename,
            'column_details': analyzer.get_detailed_column_analysis()
        }
        
        return clean_json_response(response)
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500


def load_dataframe_enhanced(file_path):
    """Enhanced dataframe loading with better header detection"""
    try:
        print(f"Attempting to load file: {file_path}")
        
        if file_path.endswith('.csv'):
            # Try different encodings and delimiter detection for CSV files
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            delimiters = [',', ';', '\t', '|']
            
            for encoding in encodings:
                for delimiter in delimiters:
                    try:
                        print(f"Trying encoding: {encoding}, delimiter: '{delimiter}'")
                        df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
                        
                        # Check if we got reasonable data
                        if len(df.columns) > 1 and len(df) > 0:
                            print(f"Successfully loaded CSV with {encoding} encoding and '{delimiter}' delimiter. Shape: {df.shape}")
                            return df
                    except Exception as e:
                        print(f"Failed with encoding {encoding} and delimiter '{delimiter}': {str(e)}")
                        continue
            
            # Fallback: try pandas' built-in CSV sniffer
            try:
                df = pd.read_csv(file_path, encoding='utf-8', engine='python')
                print(f"Loaded CSV with auto-detection. Shape: {df.shape}")
                return df
            except Exception as e:
                print(f"Auto-detection failed: {str(e)}")
                return None
                
        elif file_path.endswith(('.xlsx', '.xls')):
            try:
                # Try reading Excel with different options
                if file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path, engine='openpyxl')
                else:
                    df = pd.read_excel(file_path, engine='xlrd')
                
                print(f"Successfully loaded Excel file. Shape: {df.shape}")
                return df
            except Exception as e:
                print(f"Excel loading failed: {str(e)}")
                # Try reading all sheets and use the first non-empty one
                try:
                    excel_file = pd.ExcelFile(file_path)
                    for sheet_name in excel_file.sheet_names:
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        if not df.empty:
                            print(f"Loaded Excel sheet '{sheet_name}'. Shape: {df.shape}")
                            return df
                except Exception as e2:
                    print(f"Final Excel attempt failed: {str(e2)}")
                    return None
        
        print("File extension not supported")
        return None
        
    except Exception as e:
        print(f"General error loading file: {str(e)}")
        return None


def generate_enhanced_eda_report(analyzer):
    """Generate enhanced EDA report with proper column names"""
    return {
        'basic_info': analyzer.get_enhanced_basic_info(),
        'missing_values': analyzer.analyze_missing_values(),
        'statistical_summary': analyzer.get_statistical_summary(),
        'correlations': analyzer.analyze_correlations(),
        'distributions': analyzer.analyze_distributions(),
        'outliers': analyzer.detect_outliers(),
        'categorical_analysis': analyzer.analyze_categorical_data(),
        'column_details': analyzer.get_detailed_column_analysis()
    }


def generate_enhanced_insights(analyzer):
    """Generate enhanced insights with proper column names"""
    insights = []
    
    # Dataset overview insights
    info = analyzer.get_enhanced_basic_info()
    insights.append({
        'title': 'Dataset Overview',
        'description': f'Your dataset contains {info["shape"][0]:,} rows and {info["shape"][1]} columns with {info["numeric_columns"]} numeric, {info["categorical_columns"]} categorical, and {info["datetime_columns"]} datetime variables.',
        'type': 'info'
    })
    
    # Column mapping insights
    if info.get('column_mapping'):
        mapped_cols = [old for old, new in info['column_mapping'].items() if old != new]
        if mapped_cols:
            insights.append({
                'title': 'Column Names Enhanced',
                'description': f'Improved {len(mapped_cols)} column names from generic "Unnamed" to meaningful names based on data patterns.',
                'type': 'success'
            })
    
    # Data quality insights
    column_details = analyzer.get_detailed_column_analysis()
    high_missing_cols = [col for col, details in column_details.items() 
                        if details.get('missing_percentage', 0) > 50]
    
    if high_missing_cols:
        insights.append({
            'title': 'Data Quality Alert',
            'description': f'Found {len(high_missing_cols)} columns with >50% missing data: {", ".join(high_missing_cols[:3])}{"..." if len(high_missing_cols) > 3 else ""}',
            'type': 'warning'
        })
    
    # Correlation insights
    correlations = analyzer.analyze_correlations()
    if correlations.get('strong_correlations'):
        insights.append({
            'title': 'Strong Correlations Detected',
            'description': f'Found {len(correlations["strong_correlations"])} pairs of strongly correlated variables (|r| > 0.7).',
            'type': 'insight',
            'details': correlations['strong_correlations'][:3]
        })
    
    # Categorical insights
    categorical_analysis = analyzer.analyze_categorical_data()
    high_cardinality_cols = [col for col, analysis in categorical_analysis.items() 
                            if analysis.get('unique_values', 0) > 50]
    
    if high_cardinality_cols:
        insights.append({
            'title': 'High Cardinality Categories',
            'description': f'Found {len(high_cardinality_cols)} categorical columns with >50 unique values, which may need grouping for analysis.',
            'type': 'info'
        })
    
    # Outlier insights
    outliers = analyzer.detect_outliers()
    outlier_columns = [col for col, data in outliers.items() if data['count'] > 0]
    if outlier_columns:
        total_outliers = sum(data['count'] for data in outliers.values())
        insights.append({
            'title': 'Outliers Detected',
            'description': f'Found {total_outliers} outliers across {len(outlier_columns)} numeric columns. Review these for data quality or interesting patterns.',
            'type': 'warning'
        })
    
    return insights


def generate_enhanced_kpis(analyzer):
    """Generate enhanced KPIs with proper column information"""
    kpis = []
    
    info = analyzer.get_enhanced_basic_info()
    
    kpis.append({
        'title': 'Total Records',
        'value': f"{info['shape'][0]:,}",
        'icon': 'database'
    })
    
    kpis.append({
        'title': 'Total Columns',
        'value': info['shape'][1],
        'icon': 'columns'
    })
    
    kpis.append({
        'title': 'Numeric Columns',
        'value': info['numeric_columns'],
        'icon': 'calculator'
    })
    
    kpis.append({
        'title': 'Categorical Columns',
        'value': info['categorical_columns'],
        'icon': 'tags'
    })
    
    if info['datetime_columns'] > 0:
        kpis.append({
            'title': 'DateTime Columns',
            'value': info['datetime_columns'],
            'icon': 'calendar'
        })
    
    # Data completeness score
    missing_info = analyzer.analyze_missing_values()
    total_cells = info['shape'][0] * info['shape'][1]
    completeness_score = int(((total_cells - missing_info['total_missing']) / total_cells) * 100) if total_cells > 0 else 100
    
    kpis.append({
        'title': 'Data Completeness',
        'value': f'{completeness_score}%',
        'icon': 'check-circle'
    })
    
    # Data diversity (average unique values per column)
    column_details = analyzer.get_detailed_column_analysis()
    avg_unique = np.mean([details.get('unique_values', 0) for details in column_details.values()])
    
    kpis.append({
        'title': 'Avg Unique Values',
        'value': f"{avg_unique:.0f}",
        'icon': 'chart-bar'
    })
    
    return kpis
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}

def load_dataframe(file_path):
    """Load dataframe with better error handling and encoding detection"""
    try:
        print(f"Attempting to load file: {file_path}")
        
        if file_path.endswith('.csv'):
            # Try different encodings for CSV files
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    print(f"Trying encoding: {encoding}")
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"Successfully loaded CSV with {encoding} encoding. Shape: {df.shape}")
                    return df
                except UnicodeDecodeError:
                    print(f"Failed with encoding: {encoding}")
                    continue
                except Exception as e:
                    print(f"Error with encoding {encoding}: {str(e)}")
                    continue
            
            # If all encodings fail, try with error handling
            try:
                df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
                print(f"Loaded CSV with error handling. Shape: {df.shape}")
                return df
            except Exception as e:
                print(f"Final CSV attempt failed: {str(e)}")
                return None
                
        elif file_path.endswith(('.xlsx', '.xls')):
            try:
                # For Excel files, try different engines
                if file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path, engine='openpyxl')
                else:
                    df = pd.read_excel(file_path, engine='xlrd')
                
                print(f"Successfully loaded Excel file. Shape: {df.shape}")
                return df
            except Exception as e:
                print(f"Excel loading failed: {str(e)}")
                # Try with alternate engine
                try:
                    df = pd.read_excel(file_path)
                    print(f"Loaded Excel with default engine. Shape: {df.shape}")
                    return df
                except Exception as e2:
                    print(f"Final Excel attempt failed: {str(e2)}")
                    return None
        
        print("File extension not supported")
        return None
        
    except Exception as e:
        print(f"General error loading file: {str(e)}")
        return None

def generate_eda_report(analyzer):
    return {
        'basic_info': analyzer.get_basic_info(),
        'missing_values': analyzer.analyze_missing_values(),
        'statistical_summary': analyzer.get_statistical_summary(),
        'correlations': analyzer.analyze_correlations(),
        'distributions': analyzer.analyze_distributions(),
        'outliers': analyzer.detect_outliers(),
        'categorical_analysis': analyzer.analyze_categorical_data()
    }

def generate_insights(analyzer):
    insights = []
    
    # Dataset overview insights
    info = analyzer.get_basic_info()
    insights.append({
        'title': 'Dataset Overview',
        'description': f'Your dataset contains {info["shape"][0]:,} rows and {info["shape"][1]} columns with {info["numeric_columns"]} numeric and {info["categorical_columns"]} categorical variables.',
        'type': 'info'
    })
    
    # Missing data insights
    missing_info = analyzer.analyze_missing_values()
    if missing_info['total_missing'] > 0:
        insights.append({
            'title': 'Missing Data Detected',
            'description': f'Found {missing_info["total_missing"]} missing values across {len(missing_info["missing_by_column"])} columns. Data cleaning has been automatically applied.',
            'type': 'warning'
        })
    
    # Correlation insights
    correlations = analyzer.analyze_correlations()
    if correlations.get('strong_correlations'):
        insights.append({
            'title': 'Strong Correlations Found',
            'description': f'Discovered {len(correlations["strong_correlations"])} pairs of strongly correlated variables (|r| > 0.7).',
            'type': 'insight',
            'details': correlations['strong_correlations'][:3]
        })
    
    # Outlier insights
    outliers = analyzer.detect_outliers()
    outlier_columns = [col for col, data in outliers.items() if data['count'] > 0]
    if outlier_columns:
        insights.append({
            'title': 'Outliers Detected',
            'description': f'Found outliers in {len(outlier_columns)} numeric columns. These may represent data errors or interesting edge cases.',
            'type': 'warning'
        })
    
    return insights

def generate_kpis(analyzer):
    kpis = []
    
    info = analyzer.get_basic_info()
    
    kpis.append({
        'title': 'Total Records',
        'value': info['shape'][0],
        'icon': 'database'
    })
    
    kpis.append({
        'title': 'Total Columns',
        'value': info['shape'][1],
        'icon': 'columns'
    })
    
    kpis.append({
        'title': 'Numeric Columns',
        'value': info['numeric_columns'],
        'icon': 'calculator'
    })
    
    # Data quality score
    missing_info = analyzer.analyze_missing_values()
    total_cells = info['shape'][0] * info['shape'][1]
    quality_score = int(((total_cells - missing_info['total_missing']) / total_cells) * 100) if total_cells > 0 else 100
    
    kpis.append({
        'title': 'Data Quality Score',
        'value': f'{quality_score}%',
        'icon': 'chart-line'
    })
    
    return kpis

def clean_json_response(response):
    def clean_json(obj):
        if isinstance(obj, dict):
            return {str(k): clean_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_json(v) for v in obj]
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return obj

    cleaned_response = clean_json(response)
    
    return app.response_class(
        response=json.dumps(cleaned_response, default=str),
        mimetype='application/json'
    )

@app.route('/download_report/<filename>')
def download_report(filename):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    report_path = f"reports/{filename}"
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True)
    else:
        return jsonify({'error': 'Report not found'}), 404

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)