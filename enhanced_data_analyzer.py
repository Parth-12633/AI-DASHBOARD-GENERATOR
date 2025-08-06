import pandas as pd
import numpy as np
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataAnalyzer:
    def __init__(self, df):
        self.original_df = df.copy()
        self.df = None
        self.cleaned_df = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.column_mapping = {}
        self._preprocess_and_analyze()

    def _preprocess_and_analyze(self):
        """Preprocess the data to handle unnamed columns and detect proper structure"""
        self.df = self.original_df.copy()
        
        # Handle headers and unnamed columns
        self._fix_headers()
        
        # Analyze data types after fixing headers
        self._analyze_data_types()

    def _fix_headers(self):
        """Fix headers and handle unnamed columns intelligently"""
        print("Original columns:", list(self.df.columns))
        
        # Check if first row might be headers
        if self._first_row_looks_like_headers():
            print("Using first row as headers")
            # Use first row as headers
            new_headers = []
            for i, val in enumerate(self.df.iloc[0]):
                if pd.isna(val) or str(val).strip() == '':
                    new_headers.append(f'Column_{i+1}')
                else:
                    new_headers.append(str(val).strip())
            
            self.df.columns = new_headers
            self.df = self.df.iloc[1:].reset_index(drop=True)
        
        # Clean up column names
        self._clean_column_names()
        
        # Final check for completely empty columns
        self._remove_empty_columns()

    def _first_row_looks_like_headers(self):
        """Heuristic to determine if first row contains headers"""
        if len(self.df) < 2:
            return False
        
        first_row = self.df.iloc[0]
        second_row = self.df.iloc[1]
        
        # Check if first row has more string values
        first_row_strings = sum(1 for val in first_row if isinstance(val, str) and not str(val).replace('.', '').replace('-', '').isdigit())
        second_row_strings = sum(1 for val in second_row if isinstance(val, str) and not str(val).replace('.', '').replace('-', '').isdigit())
        
        # Check if first row values look like column names
        header_like_patterns = ['name', 'id', 'date', 'time', 'value', 'amount', 'price', 'count', 'total', 'score']
        first_row_header_like = sum(1 for val in first_row if any(pattern in str(val).lower() for pattern in header_like_patterns))
        
        return (first_row_strings > len(first_row) * 0.5) or (first_row_header_like > 0)

    def _clean_column_names(self):
        """Clean and standardize column names"""
        new_columns = []
        column_counts = {}
        
        for i, col in enumerate(self.df.columns):
            col_str = str(col).strip()
            
            # Handle unnamed columns
            if col_str.startswith('Unnamed:') or col_str == '' or col_str == 'nan':
                # Try to infer column name from data
                col_name = self._infer_column_name(i)
                if not col_name:
                    col_name = f'Column_{i+1}'
            else:
                col_name = col_str
            
            # Clean the name
            col_name = self._sanitize_column_name(col_name)
            
            # Handle duplicates
            if col_name in column_counts:
                column_counts[col_name] += 1
                col_name = f"{col_name}_{column_counts[col_name]}"
            else:
                column_counts[col_name] = 0
            
            new_columns.append(col_name)
            self.column_mapping[self.df.columns[i]] = col_name
        
        self.df.columns = new_columns
        print("New columns:", list(self.df.columns))

    def _infer_column_name(self, col_index):
        """Infer column name from data patterns"""
        if col_index >= len(self.df.columns):
            return None
        
        col_data = self.df.iloc[:, col_index].dropna()
        if len(col_data) == 0:
            return None
        
        # Sample first few values
        sample_data = col_data.head(10).astype(str)
        
        # Check for common patterns
        if sample_data.str.contains('@').any():
            return 'Email'
        elif sample_data.str.match(r'^\d{4}-\d{2}-\d{2}').any():
            return 'Date'
        elif sample_data.str.match(r'^\d+$').all():
            return 'ID' if col_index == 0 else 'Number'
        elif sample_data.str.match(r'^\d+\.\d+$').any():
            return 'Value'
        elif any(word in ' '.join(sample_data.str.lower()) for word in ['male', 'female', 'yes', 'no', 'true', 'false']):
            return 'Category'
        else:
            return None

    def _sanitize_column_name(self, name):
        """Sanitize column name"""
        # Remove special characters and replace with underscore
        import re
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', '_', name)
        name = name.strip('_')
        
        # Capitalize first letter of each word
        name = '_'.join(word.capitalize() for word in name.split('_'))
        
        return name if name else 'Unknown'

    def _remove_empty_columns(self):
        """Remove completely empty columns"""
        non_empty_columns = []
        for col in self.df.columns:
            if not self.df[col].dropna().empty:
                non_empty_columns.append(col)
        
        if len(non_empty_columns) < len(self.df.columns):
            print(f"Removing {len(self.df.columns) - len(non_empty_columns)} empty columns")
            self.df = self.df[non_empty_columns]

    def _analyze_data_types(self):
        """Analyze and categorize column data types with better detection"""
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        
        for col in self.df.columns:
            col_data = self.df[col].dropna()
            if len(col_data) == 0:
                continue
            
            # Try to convert to numeric
            if self._is_numeric_column(col_data):
                # Convert to numeric
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.numeric_columns.append(col)
            elif self._is_datetime_column(col_data):
                # Convert to datetime
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    self.datetime_columns.append(col)
                except:
                    self.categorical_columns.append(col)
            else:
                self.categorical_columns.append(col)

    def _is_numeric_column(self, col_data):
        """Check if column should be treated as numeric"""
        # Try converting sample to numeric
        sample = col_data.head(100).astype(str)
        numeric_count = 0
        
        for val in sample:
            try:
                float(val.replace(',', '').replace('$', '').replace('%', ''))
                numeric_count += 1
            except:
                continue
        
        return numeric_count > len(sample) * 0.7

    def _is_datetime_column(self, col_data):
        """Check if column should be treated as datetime"""
        sample = col_data.head(50).astype(str)
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}-\d{2}-\d{4}',
            r'\d{4}/\d{2}/\d{2}',
            r'\w+\s+\d{1,2},\s+\d{4}'
        ]
        
        for pattern in datetime_patterns:
            import re
            if sample.str.contains(pattern, regex=True).sum() > len(sample) * 0.5:
                return True
        
        return False

    def generate_smart_visualizations(self):
        import json
        import numpy as np
        import plotly.graph_objects as go
        from plotly.utils import PlotlyJSONEncoder

        df = self.cleaned_df if self.cleaned_df is not None else self.df
        charts = []

        # 1. Trend/Line Chart (numeric over time)
        if self.datetime_columns and self.numeric_columns:
            for num_col in self.numeric_columns[:2]:
                dt_col = self.datetime_columns[0]
                temp_df = df[[dt_col, num_col]].dropna().sort_values(dt_col)
                if not temp_df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=temp_df[dt_col],
                        y=temp_df[num_col],
                        mode='lines+markers'
                    ))
                    fig.update_layout(
                        title=f"{num_col.replace('_', ' ').title()} Trend Over {dt_col.replace('_', ' ').title()}",
                        xaxis_title=dt_col.replace('_', ' ').title(),
                        yaxis_title=num_col.replace('_', ' ').title(),
                        template="plotly_dark"
                    )
                    charts.append({
                        'type': 'line',
                        'title': f"{num_col.replace('_', ' ').title()} Trend Over {dt_col.replace('_', ' ').title()}",
                        'data': json.dumps(fig, cls=PlotlyJSONEncoder)
                    })

        # 2. Bar Chart (top categories)
        for cat_col in self.categorical_columns[:2]:
            value_counts = df[cat_col].value_counts().head(8)
            if not value_counts.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=value_counts.index.astype(str),
                    y=value_counts.values
                ))
                fig.update_layout(
                    title=f"Top {cat_col.replace('_', ' ').title()} Categories",
                    xaxis_title=cat_col.replace('_', ' ').title(),
                    yaxis_title="Count",
                    template="plotly_dark"
                )
                charts.append({
                    'type': 'bar',
                    'title': f"Top {cat_col.replace('_', ' ').title()} Categories",
                    'data': json.dumps(fig, cls=PlotlyJSONEncoder)
                })

        # 3. Donut/Pie Chart (if categorical with 2-5 values)
        for cat_col in self.categorical_columns[:2]:
            value_counts = df[cat_col].value_counts()
            if 2 <= len(value_counts) <= 5:
                fig = go.Figure()
                fig.add_trace(go.Pie(
                    labels=value_counts.index.astype(str),
                    values=value_counts.values,
                    hole=0.5
                ))
                fig.update_layout(
                    title=f"{cat_col.replace('_', ' ').title()} Distribution",
                    template="plotly_dark"
                )
                charts.append({
                    'type': 'donut',
                    'title': f"{cat_col.replace('_', ' ').title()} Distribution",
                    'data': json.dumps(fig, cls=PlotlyJSONEncoder)
                })

        # 4. Stacked Bar (if two categoricals)
        if len(self.categorical_columns) >= 2:
            cat1, cat2 = self.categorical_columns[:2]
            cross = df.groupby([cat1, cat2]).size().unstack(fill_value=0)
            if not cross.empty:
                fig = go.Figure()
                for col in cross.columns:
                    fig.add_trace(go.Bar(
                        x=cross.index.astype(str),
                        y=cross[col],
                        name=str(col)
                    ))
                fig.update_layout(
                    barmode='stack',
                    title=f"{cat1.replace('_', ' ').title()} by {cat2.replace('_', ' ').title()}",
                    xaxis_title=cat1.replace('_', ' ').title(),
                    yaxis_title="Count",
                    template="plotly_dark"
                )
                charts.append({
                    'type': 'stacked_bar',
                    'title': f"{cat1.replace('_', ' ').title()} by {cat2.replace('_', ' ').title()}",
                    'data': json.dumps(fig, cls=PlotlyJSONEncoder)
                })

        # 5. Correlation Heatmap (if >1 numeric)
        if len(self.numeric_columns) > 1:
            corr = df[self.numeric_columns].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=[col.replace('_', ' ').title() for col in corr.columns],
                y=[col.replace('_', ' ').title() for col in corr.columns],
                colorscale='RdBu',
                zmid=0
            ))
            fig.update_layout(
                title="Correlation Heatmap of Numeric Columns",
                template="plotly_dark"
            )
            charts.append({
                'type': 'heatmap',
                'title': "Correlation Heatmap of Numeric Columns",
                'data': json.dumps(fig, cls=PlotlyJSONEncoder)
            })

        # 6. Scatter for strong correlations
        if len(self.numeric_columns) > 1:
            corr = df[self.numeric_columns].corr()
            pairs = []
            for i, col1 in enumerate(self.numeric_columns):
                for j, col2 in enumerate(self.numeric_columns):
                    if i < j and abs(corr.iloc[i, j]) > 0.7:
                        pairs.append((col1, col2, corr.iloc[i, j]))
            for col1, col2, corr_val in pairs[:2]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df[col1], y=df[col2], mode='markers',
                    marker=dict(opacity=0.6),
                    name=f'r = {corr_val:.2f}'
                ))
                fig.update_layout(
                    title=f"{col1.replace('_', ' ').title()} vs {col2.replace('_', ' ').title()} (r={corr_val:.2f})",
                    xaxis_title=col1.replace('_', ' ').title(),
                    yaxis_title=col2.replace('_', ' ').title(),
                    template="plotly_dark"
                )
                charts.append({
                    'type': 'scatter',
                    'title': f"{col1.replace('_', ' ').title()} vs {col2.replace('_', ' ').title()} (r={corr_val:.2f})",
                    'data': json.dumps(fig, cls=PlotlyJSONEncoder)
                })

        # 7. Box/Violin for numeric columns (outlier/variance)
        for num_col in self.numeric_columns[:2]:
            data = df[num_col].dropna()
            if not data.empty:
                fig = go.Figure()
                fig.add_trace(go.Box(y=data, name=num_col))
                fig.update_layout(
                    title=f"{num_col.replace('_', ' ').title()} Distribution (Box Plot)",
                    yaxis_title=num_col.replace('_', ' ').title(),
                    template="plotly_dark"
                )
                charts.append({
                    'type': 'box',
                    'title': f"{num_col.replace('_', ' ').title()} Distribution (Box Plot)",
                    'data': json.dumps(fig, cls=PlotlyJSONEncoder)
                })

        return charts

    def get_enhanced_basic_info(self):
        """Get enhanced basic information about the dataset"""
        df = self.cleaned_df if self.cleaned_df is not None else self.df
        
        return {
            'shape': df.shape,
            'memory_usage': int(df.memory_usage(deep=True).sum()),
            'dtypes': {str(k): str(v) for k, v in df.dtypes.items()},
            'numeric_columns': len(self.numeric_columns),
            'categorical_columns': len(self.categorical_columns),
            'datetime_columns': len(self.datetime_columns),
            'column_details': {
                'numeric': self.numeric_columns,
                'categorical': self.categorical_columns,
                'datetime': self.datetime_columns
            },
            'column_mapping': self.column_mapping
        }

    def get_detailed_column_analysis(self):
        """Get detailed analysis for each column"""
        df = self.cleaned_df if self.cleaned_df is not None else self.df
        analysis = {}
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            if col in self.numeric_columns:
                analysis[col] = {
                    'type': 'numeric',
                    'count': len(col_data),
                    'mean': float(col_data.mean()) if len(col_data) > 0 else None,
                    'median': float(col_data.median()) if len(col_data) > 0 else None,
                    'std': float(col_data.std()) if len(col_data) > 0 else None,
                    'min': float(col_data.min()) if len(col_data) > 0 else None,
                    'max': float(col_data.max()) if len(col_data) > 0 else None,
                    'unique_values': int(col_data.nunique()),
                    'missing_percentage': float((df[col].isna().sum() / len(df)) * 100)
                }
            elif col in self.categorical_columns:
                value_counts = col_data.value_counts()
                analysis[col] = {
                    'type': 'categorical',
                    'count': len(col_data),
                    'unique_values': int(col_data.nunique()),
                    'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'missing_percentage': float((df[col].isna().sum() / len(df)) * 100),
                    'top_values': {str(k): int(v) for k, v in value_counts.head(5).items()}
                }
            elif col in self.datetime_columns:
                col_data = pd.to_datetime(col_data, errors='coerce').dropna()
                analysis[col] = {
                    'type': 'datetime',
                    'count': len(col_data),
                    'min_date': str(col_data.min()) if len(col_data) > 0 else None,
                    'max_date': str(col_data.max()) if len(col_data) > 0 else None,
                    'unique_values': int(col_data.nunique()),
                    'missing_percentage': float((df[col].isna().sum() / len(df)) * 100)
                }
        
        return analysis

    # Include all other methods from the original DataAnalyzer class
    def clean_data(self, cleaning_options=None):
        """Clean the dataset based on provided options"""
        if cleaning_options is None:
            cleaning_options = {
                'remove_duplicates': True,
                'handle_missing': 'auto',
                'remove_outliers': False,
                'standardize_text': True
            }

        self.cleaned_df = self.df.copy()
        report = {
            'original_shape': self.df.shape,
            'steps_performed': [],
            'missing_values_before': int(self.df.isnull().sum().sum()),
            'duplicates_before': int(self.df.duplicated().sum())
        }

        # Remove duplicates
        if cleaning_options.get('remove_duplicates', True):
            before = len(self.cleaned_df)
            self.cleaned_df.drop_duplicates(inplace=True)
            after = len(self.cleaned_df)
            if before != after:
                report['steps_performed'].append(f"Removed {before - after} duplicate rows")

        # Handle missing values intelligently
        strategy = cleaning_options.get('handle_missing', 'auto')
        for col in self.cleaned_df.columns:
            missing_count = self.cleaned_df[col].isnull().sum()
            if missing_count > 0:
                if col in self.numeric_columns:
                    # Use median for numeric columns
                    fill_value = self.cleaned_df[col].median()
                    self.cleaned_df[col].fillna(fill_value, inplace=True)
                    report['steps_performed'].append(f"Filled {missing_count} missing values in '{col}' with median ({fill_value:.2f})")
                elif col in self.categorical_columns:
                    # Use mode for categorical columns
                    mode_val = self.cleaned_df[col].mode()
                    fill_value = mode_val[0] if not mode_val.empty else 'Unknown'
                    self.cleaned_df[col].fillna(fill_value, inplace=True)
                    report['steps_performed'].append(f"Filled {missing_count} missing values in '{col}' with mode ('{fill_value}')")
                else:
                    # For datetime or other columns
                    fill_value = 'Unknown'
                    self.cleaned_df[col].fillna(fill_value, inplace=True)
                    report['steps_performed'].append(f"Filled {missing_count} missing values in '{col}' with '{fill_value}'")

        # Standardize text columns
        if cleaning_options.get('standardize_text', True):
            for col in self.categorical_columns:
                if self.cleaned_df[col].dtype == 'object':
                    self.cleaned_df[col] = self.cleaned_df[col].astype(str).str.strip()
            report['steps_performed'].append("Standardized text columns")

        report['final_shape'] = self.cleaned_df.shape
        report['missing_values_after'] = int(self.cleaned_df.isnull().sum().sum())
        
        return self.cleaned_df, report

    def analyze_missing_values(self):
        """Analyze missing values in the dataset"""
        df = self.cleaned_df if self.cleaned_df is not None else self.df
        
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        return {
            'total_missing': int(missing_data.sum()),
            'missing_by_column': {str(k): int(v) for k, v in missing_data[missing_data > 0].items()},
            'missing_percentage': {str(k): float(v) for k, v in missing_percent[missing_percent > 0].items()}
        }

    def get_statistical_summary(self):
        """Get statistical summary for numeric columns"""
        df = self.cleaned_df if self.cleaned_df is not None else self.df
        
        if not self.numeric_columns:
            return {}
        
        summary = df[self.numeric_columns].describe()
        return {col: {stat: float(val) for stat, val in summary[col].items()} 
                for col in summary.columns}

    def analyze_correlations(self):
        """Analyze correlations between numeric variables"""
        df = self.cleaned_df if self.cleaned_df is not None else self.df
        
        if len(self.numeric_columns) < 2:
            return {'correlation_matrix': {}, 'strong_correlations': []}
        
        corr = df[self.numeric_columns].corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                val = corr.iloc[i, j]
                if not np.isnan(val) and abs(val) > 0.7:
                    strong_correlations.append({
                        'var1': str(corr.columns[i]),
                        'var2': str(corr.columns[j]),
                        'correlation': round(float(val), 3)
                    })
        
        # Convert correlation matrix to JSON-serializable format
        correlation_matrix = {}
        for col in corr.columns:
            correlation_matrix[str(col)] = {}
            for row in corr.index:
                val = corr.loc[row, col]
                correlation_matrix[str(col)][str(row)] = float(val) if not np.isnan(val) else None
        
        return {
            'correlation_matrix': correlation_matrix,
            'strong_correlations': strong_correlations
        }

    def analyze_distributions(self):
        """Analyze distributions of numeric variables"""
        df = self.cleaned_df if self.cleaned_df is not None else self.df
        
        distributions = {}
        for col in self.numeric_columns:
            data = df[col].dropna()
            if data.empty or len(data) < 2:
                continue
            
            try:
                distributions[str(col)] = {
                    'skewness': float(data.skew()),
                    'kurtosis': float(data.kurtosis()),
                    'is_normal': abs(data.skew()) < 0.5 and abs(data.kurtosis()) < 3,
                    'unique_values': int(data.nunique()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()) if data.std() is not np.nan else 0.0
                }
            except:
                continue
        
        return distributions

    def detect_outliers(self):
        """Detect outliers using IQR method"""
        df = self.cleaned_df if self.cleaned_df is not None else self.df
        
        outliers = {}
        for col in self.numeric_columns:
            data = df[col].dropna()
            if data.empty or len(data) < 4:
                continue
            
            try:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (data < lower_bound) | (data > upper_bound)
                outlier_count = int(outlier_mask.sum())
                
                outliers[str(col)] = {
                    'count': outlier_count,
                    'percentage': float(outlier_count / len(data) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
            except:
                continue
        
        return outliers

    def analyze_categorical_data(self):
        """Analyze categorical variables"""
        df = self.cleaned_df if self.cleaned_df is not None else self.df
        
        categorical_analysis = {}
        for col in self.categorical_columns:
            data = df[col].dropna()
            if data.empty:
                continue
            
            try:
                value_counts = data.value_counts()
                categorical_analysis[str(col)] = {
                    'unique_values': int(data.nunique()),
                    'most_frequent': str(value_counts.index[0]) if not value_counts.empty else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if not value_counts.empty else 0,
                    'value_counts': {str(k): int(v) for k, v in value_counts.head(10).items()}
                }
            except:
                continue
        
        return categorical_analysis

    def export_dashboard_and_eda_to_pdf(self, charts, eda_report, filename='dashboard_report.pdf'):
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.pagesizes import A4
        import io
        import plotly.graph_objects as go
        import json
        import os

        pdf_path = f"reports/{filename}"
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        story.append(Paragraph("Auto-Generated Dashboard & EDA Report", styles['Title']))
        story.append(Spacer(1, 12))

        # KPIs and EDA summary
        story.append(Paragraph("Basic Info", styles['Heading2']))
        for k, v in eda_report['basic_info'].items():
            story.append(Paragraph(f"<b>{k.replace('_',' ').title()}:</b> {v}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Add charts
        for chart in charts:
            story.append(Paragraph(chart['title'], styles['Heading3']))
            fig_json = json.loads(chart['data'])
            fig = go.Figure(fig_json)
            img_bytes = fig.to_image(format="png")
            img_stream = io.BytesIO(img_bytes)
            img = RLImage(img_stream, width=400, height=250)
            story.append(img)
            story.append(Spacer(1, 20))

        # EDA details
        story.append(Paragraph("EDA Details", styles['Heading2']))
        for section, content in eda_report.items():
            if section == 'basic_info':
                continue
            story.append(Paragraph(f"<b>{section.replace('_',' ').title()}:</b>", styles['Heading3']))
            story.append(Paragraph(str(content), styles['Normal']))
            story.append(Spacer(1, 8))

        doc.build(story)
        return pdf_path