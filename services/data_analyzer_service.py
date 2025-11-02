import os
from enhanced_data_analyzer import EnhancedDataAnalyzer
from utils.helpers import load_dataframe_enhanced, generate_enhanced_eda_report, generate_enhanced_insights

def process_file(file_path):
    """Process uploaded file and return analysis results"""
    try:
        print(f"Processing file: {file_path}")
        
        # Load dataframe
        df = load_dataframe_enhanced(file_path)
        if df is None:
            return {'error': 'Could not read the file. Please ensure it\'s a valid CSV or Excel file.'}
        
        print(f"DataFrame loaded successfully. Shape: {df.shape}")
        
        # Initialize enhanced analyzer
        analyzer = EnhancedDataAnalyzer(df)
        
        print("Step 1: Analyzer initialized")
        # Clean data
        cleaned_df, cleaning_report = analyzer.clean_data()
        print("Step 2: Data cleaned")
        # Generate enhanced EDA report
        eda_report = generate_enhanced_eda_report(analyzer)
        print("Step 3: EDA report generated")
        # Generate dual-recommended visualizations (ML + Gemini)
        charts = analyzer.generate_dual_recommended_visualizations(cleaned_df)
        print("Step 4: Dual-recommended charts generated")
        # Generate enhanced insights
        insights = generate_enhanced_insights(analyzer)
        print("Step 5: Insights generated")
        # Generate enhanced KPIs using smart KPI generation
        kpis = analyzer.generate_smart_kpis(cleaned_df)
        print("Step 6: Smart KPIs generated")
        # Generate PDF report
        pdf_filename = f"{os.path.basename(file_path).split('.')[0]}_dashboard_report.pdf"
        pdf_path = analyzer.export_dashboard_and_eda_to_pdf(charts, eda_report, filename=pdf_filename)
        print("Step 7: PDF generated")
        
        # Prepare response
        response = {
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
        
        return response
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': f'Error processing file: {str(e)}'}
