import pandas as pd
import numpy as np
import json
from flask import jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import os

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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}

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
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return obj

    cleaned_response = clean_json(response)
    
    return jsonify(cleaned_response)
