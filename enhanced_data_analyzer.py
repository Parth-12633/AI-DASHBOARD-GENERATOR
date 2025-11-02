import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json
import os
import google.generativeai as genai
from typing import Dict, List, Any, Tuple

class EnhancedDataAnalyzer:
    def __init__(self, df):
        self.df = df
        self.cleaned_df = None
        self.column_mapping = {}
        
        # Initialize Gemini API
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
        else:
            self.gemini_model = None
            print("Warning: GEMINI_API_KEY not found. Using ML-only recommendations.")
        
        # Detect column types
        self._detect_column_types()
        
        # Clean column names
        self._clean_column_names()

    def _detect_column_types(self):
        """Detect numeric, categorical, and datetime columns"""
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        
        for col in self.df.columns:
            # Skip if all values are null
            if self.df[col].isna().all():
                continue
                
            # Try to convert to numeric
            numeric_series = pd.to_numeric(self.df[col], errors='coerce')
            if not numeric_series.isna().all():
                # Check if most values are numeric
                numeric_ratio = (~numeric_series.isna()).sum() / len(self.df[col])
                if numeric_ratio > 0.7:
                    self.numeric_columns.append(col)
                    continue
            
            # Try to convert to datetime
            datetime_series = pd.to_datetime(self.df[col], errors='coerce')
            if not datetime_series.isna().all():
                # Check if most values are datetime
                datetime_ratio = (~datetime_series.isna()).sum() / len(self.df[col])
                if datetime_ratio > 0.7:
                    self.datetime_columns.append(col)
                    continue
            
            # Otherwise, treat as categorical
            self.categorical_columns.append(col)

    def _clean_column_names(self):
        """Clean and standardize column names"""
        for i, col in enumerate(self.df.columns):
            if col.startswith('Unnamed') or col == '':
                # Try to infer name from data
                sample_values = self.df[col].dropna().head(3).tolist()
                if sample_values:
                    new_name = f"Column_{i+1}_{str(sample_values[0])[:20]}"
                else:
                    new_name = f"Column_{i+1}"
                self.column_mapping[col] = new_name
                self.df = self.df.rename(columns={col: new_name})
            else:
                # Clean existing names
                clean_name = col.strip().replace(' ', '_').replace('-', '_')
                if clean_name != col:
                    self.column_mapping[col] = clean_name
                    self.df = self.df.rename(columns={col: clean_name})

        # Update column type lists to use new column names
        self.numeric_columns = [self.column_mapping.get(col, col) for col in self.numeric_columns]
        self.categorical_columns = [self.column_mapping.get(col, col) for col in self.categorical_columns]
        self.datetime_columns = [self.column_mapping.get(col, col) for col in self.datetime_columns]

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

    def get_gemini_visualization_recommendations(self, df) -> List[Dict]:
        """Get visualization recommendations from Google Gemini AI"""
        if not self.gemini_model:
            return []
        
        try:
            # Prepare dataset summary for Gemini
            dataset_summary = self._prepare_dataset_summary_for_gemini(df)
            
            prompt = f"""
            As a data visualization expert, analyze this dataset and recommend the TOP 5 most effective visualizations for a Power BI-style business dashboard:

            Dataset Summary:
            {dataset_summary}

            Please recommend exactly 5 visualizations in order of priority, considering:
            1. Business value and insights
            2. Data relationships and patterns
            3. Power BI best practices
            4. User engagement and readability

            Return your response as a JSON array with this exact format:
            [
                {{
                    "visualization": "chart_type",
                    "title": "Chart Title",
                    "priority": 1,
                    "reasoning": "Why this chart is recommended",
                    "columns": ["column1", "column2"],
                    "business_insight": "What business insight this provides"
                }}
            ]

            Available chart types: line, bar, pie, donut, area, scatter, box, histogram, heatmap, treemap, waterfall, sankey, radar, gauge, funnel, bullet, scatter3d

            Focus on creating a cohesive dashboard that tells a data story.
            """
            
            response = self.gemini_model.generate_content(prompt)
            recommendations_text = response.text
            
            # Parse JSON response
            try:
                # Extract JSON from response
                start_idx = recommendations_text.find('[')
                end_idx = recommendations_text.rfind(']') + 1
                json_str = recommendations_text[start_idx:end_idx]
                recommendations = json.loads(json_str)
                
                # Validate and clean recommendations
                valid_recommendations = []
                for rec in recommendations:
                    if all(key in rec for key in ['visualization', 'title', 'priority', 'reasoning', 'columns']):
                        valid_recommendations.append(rec)
                
                return valid_recommendations[:5]  # Return top 5
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing Gemini response: {e}")
                return []
                
        except Exception as e:
            print(f"Error getting Gemini recommendations: {e}")
            return []

    def _prepare_dataset_summary_for_gemini(self, df) -> str:
        """Prepare a comprehensive dataset summary for Gemini AI"""
        summary = f"""
        - Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns
        - Numeric Columns ({len(self.numeric_columns)}): {', '.join(self.numeric_columns[:5])}
        - Categorical Columns ({len(self.categorical_columns)}): {', '.join(self.categorical_columns[:5])}
        - DateTime Columns ({len(self.datetime_columns)}): {', '.join(self.datetime_columns[:3])}
        """
        
        # Add column details
        if self.numeric_columns:
            summary += "\n- Numeric Column Statistics:\n"
            for col in self.numeric_columns[:3]:
                data = df[col].dropna()
                if not data.empty:
                    summary += f"  * {col}: mean={data.mean():.2f}, std={data.std():.2f}, range=[{data.min():.2f}, {data.max():.2f}]\n"
        
        if self.categorical_columns:
            summary += "\n- Categorical Column Details:\n"
            for col in self.categorical_columns[:3]:
                data = df[col].dropna()
                if not data.empty:
                    unique_count = data.nunique()
                    top_value = data.value_counts().index[0] if not data.empty else "N/A"
                    summary += f"  * {col}: {unique_count} unique values, top='{top_value}'\n"
        
        if self.datetime_columns:
            summary += "\n- DateTime Column Details:\n"
            for col in self.datetime_columns[:2]:
                data = pd.to_datetime(df[col], errors='coerce').dropna()
                if not data.empty:
                    summary += f"  * {col}: range from {data.min()} to {data.max()}\n"
        
        # Add correlations if available
        if len(self.numeric_columns) > 1:
            corr_matrix = df[self.numeric_columns].corr()
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.7:
                        strong_corr.append(f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]} (r={corr_val:.2f})")
            
            if strong_corr:
                summary += f"\n- Strong Correlations: {', '.join(strong_corr[:3])}\n"
        
        return summary

    def generate_dual_recommended_visualizations(self, df) -> List[Dict]:
        """Generate visualizations using dual recommendation system (ML + Gemini)"""
        # Get ML recommendations
        ml_scores = self._score_visualizations(df)
        ml_recommendations = sorted(ml_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get Gemini recommendations
        gemini_recommendations = self.get_gemini_visualization_recommendations(df)
        
        # Combine and rank recommendations
        combined_recommendations = self._combine_recommendations(ml_recommendations, gemini_recommendations)
        
        # Generate top 5 visualizations with Power BI styling
        charts = []
        for i, (chart_type, confidence, gemini_info) in enumerate(combined_recommendations[:5]):
            chart = self._generate_powerbi_chart(chart_type, df, gemini_info, i+1)
            if chart:
                charts.append(chart)
        
        return charts

    def _combine_recommendations(self, ml_recommendations, gemini_recommendations) -> List[Tuple]:
        """Combine ML and Gemini recommendations with weighted scoring"""
        combined = {}
        
        # Process ML recommendations (weight: 0.6)
        for chart_type, score in ml_recommendations:
            combined[chart_type] = {
                'ml_score': score * 0.6,
                'gemini_priority': 0,
                'gemini_info': None
            }
        
        # Process Gemini recommendations (weight: 0.4)
        for rec in gemini_recommendations:
            chart_type = rec['visualization']
            priority = rec.get('priority', 6)
            gemini_score = max(0, 10 - priority) * 0.4  # Higher priority = lower number = higher score
            
            if chart_type in combined:
                combined[chart_type]['gemini_priority'] = gemini_score
                combined[chart_type]['gemini_info'] = rec
            else:
                combined[chart_type] = {
                    'ml_score': 0,
                    'gemini_priority': gemini_score,
                    'gemini_info': rec
                }
        
        # Calculate final scores and sort
        final_recommendations = []
        for chart_type, scores in combined.items():
            total_score = scores['ml_score'] + scores['gemini_priority']
            final_recommendations.append((chart_type, total_score, scores['gemini_info']))
        
        return sorted(final_recommendations, key=lambda x: x[1], reverse=True)

    def _generate_powerbi_chart(self, chart_type, df, gemini_info, position) -> Dict:
        """Generate a single chart with Power BI styling"""
        try:
            if chart_type == 'line':
                return self._generate_powerbi_line_chart(df, gemini_info, position)
            elif chart_type == 'bar':
                return self._generate_powerbi_bar_chart(df, gemini_info, position)
            elif chart_type == 'pie':
                return self._generate_powerbi_pie_chart(df, gemini_info, position)
            elif chart_type == 'donut':
                return self._generate_powerbi_donut_chart(df, gemini_info, position)
            elif chart_type == 'area':
                return self._generate_powerbi_area_chart(df, gemini_info, position)
            elif chart_type == 'scatter':
                return self._generate_powerbi_scatter_chart(df, gemini_info, position)
            elif chart_type == 'heatmap':
                return self._generate_powerbi_heatmap_chart(df, gemini_info, position)
            elif chart_type == 'treemap':
                return self._generate_powerbi_treemap_chart(df, gemini_info, position)
            elif chart_type == 'gauge':
                return self._generate_powerbi_gauge_chart(df, gemini_info, position)
            else:
                # Fallback to generic chart generation
                return self._generate_generic_powerbi_chart(chart_type, df, gemini_info, position)
        except Exception as e:
            print(f"Error generating {chart_type} chart: {e}")
            return None

    def _get_powerbi_colors(self) -> List[str]:
        """Power BI default color palette"""
        return [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Olive
            '#17becf'   # Cyan
        ]

    def _get_powerbi_layout(self, title, subtitle=None) -> Dict:
        """Get Power BI style layout configuration"""
        layout = {
            'template': 'plotly_white',
            'plot_bgcolor': 'rgba(255, 255, 255, 1)',
            'paper_bgcolor': 'rgba(255, 255, 255, 1)',
            'font': {
                'family': 'Segoe UI, Arial, sans-serif',
                'size': 12,
                'color': '#323130'
            },
            'title': {
                'text': title,
                'font': {'size': 16, 'color': '#323130', 'family': 'Segoe UI, Arial, sans-serif'},
                'x': 0.05,
                'xanchor': 'left',
                'y': 0.95,
                'yanchor': 'top'
            },
            'margin': {'t': 60, 'r': 40, 'b': 60, 'l': 60},
            'xaxis': {
                'gridcolor': '#f3f2f1',
                'gridwidth': 1,
                'color': '#605e5c',
                'title_font': {'size': 12, 'color': '#323130'},
                'tickfont': {'size': 11, 'color': '#605e5c'}
            },
            'yaxis': {
                'gridcolor': '#f3f2f1',
                'gridwidth': 1,
                'color': '#605e5c',
                'title_font': {'size': 12, 'color': '#323130'},
                'tickfont': {'size': 11, 'color': '#605e5c'}
            },
            'showlegend': True,
            'legend': {
                'orientation': 'v',
                'x': 1.02,
                'y': 1,
                'font': {'size': 11, 'color': '#323130'}
            }
        }
        
        if subtitle:
            layout['annotations'] = [{
                'text': subtitle,
                'x': 0.05,
                'y': 0.88,
                'xanchor': 'left',
                'yanchor': 'top',
                'font': {'size': 11, 'color': '#605e5c'},
                'showarrow': False
            }]
        
        return layout

    def _generate_powerbi_line_chart(self, df, gemini_info, position) -> Dict:
        """Generate Power BI style line chart"""
        if not (self.datetime_columns and self.numeric_columns):
            return None
        
        dt_col = self.datetime_columns[0]
        num_col = self.numeric_columns[0]
        temp_df = df[[dt_col, num_col]].dropna().sort_values(dt_col)
        
        if temp_df.empty:
            return None
        
        colors = self._get_powerbi_colors()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=temp_df[dt_col],
            y=temp_df[num_col],
            mode='lines+markers',
            line=dict(color=colors[0], width=3),
            marker=dict(color=colors[0], size=6, line=dict(color='white', width=2)),
            name=num_col.replace('_', ' ').title(),
            hovertemplate=f'<b>{dt_col.replace("_", " ").title()}</b><br>' +
                         f'{num_col.replace("_", " ").title()}: %{{y:,.2f}}<br>' +
                         '<extra></extra>'
        ))
        
        title = gemini_info.get('title', f"{num_col.replace('_', ' ').title()} Trend") if gemini_info else f"{num_col.replace('_', ' ').title()} Trend"
        subtitle = gemini_info.get('business_insight', f"Shows the trend of {num_col.replace('_', ' ').lower()} over time") if gemini_info else None
        
        fig.update_layout(
            **self._get_powerbi_layout(title, subtitle),
            xaxis_title=dt_col.replace('_', ' ').title(),
            yaxis_title=num_col.replace('_', ' ').title()
        )
        
        return {
            'type': 'line',
            'title': title,
            'position': position,
            'data': json.dumps(fig, cls=PlotlyJSONEncoder),
            'insight': subtitle,
            'chart_type': 'line'
        }

    def _generate_powerbi_bar_chart(self, df, gemini_info, position) -> Dict:
        """Generate Power BI style bar chart"""
        if not self.categorical_columns:
            return None
        
        cat_col = self.categorical_columns[0]
        value_counts = df[cat_col].value_counts().head(10)
        
        if value_counts.empty:
            return None
        
        colors = self._get_powerbi_colors()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            marker=dict(
                color=[colors[i % len(colors)] for i in range(len(value_counts))],
                line=dict(color='white', width=1)
            ),
            hovertemplate=f'<b>%{{x}}</b><br>Count: %{{y:,}}<br><extra></extra>'
        ))
        
        title = gemini_info.get('title', f"Top {cat_col.replace('_', ' ').title()} Categories") if gemini_info else f"Top {cat_col.replace('_', ' ').title()} Categories"
        subtitle = gemini_info.get('business_insight', f"Distribution of {cat_col.replace('_', ' ').lower()} categories") if gemini_info else None
        
        fig.update_layout(
            **self._get_powerbi_layout(title, subtitle),
            xaxis_title=cat_col.replace('_', ' ').title(),
            yaxis_title="Count"
        )
        
        return {
            'type': 'bar',
            'title': title,
            'position': position,
            'data': json.dumps(fig, cls=PlotlyJSONEncoder),
            'insight': subtitle,
            'chart_type': 'bar'
        }

    def _generate_powerbi_pie_chart(self, df, gemini_info, position) -> Dict:
        """Generate Power BI style pie chart"""
        if not self.categorical_columns:
            return None
        
        cat_col = self.categorical_columns[0]
        value_counts = df[cat_col].value_counts().head(8)
        
        if len(value_counts) < 2:
            return None
        
        colors = self._get_powerbi_colors()
        
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=value_counts.index.astype(str),
            values=value_counts.values,
            marker=dict(
                colors=[colors[i % len(colors)] for i in range(len(value_counts))],
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<br><extra></extra>',
            textposition='outside',
            textinfo='label+percent'
        ))
        
        title = gemini_info.get('title', f"{cat_col.replace('_', ' ').title()} Distribution") if gemini_info else f"{cat_col.replace('_', ' ').title()} Distribution"
        subtitle = gemini_info.get('business_insight', f"Proportional breakdown of {cat_col.replace('_', ' ').lower()} categories") if gemini_info else None
        
        fig.update_layout(
            **self._get_powerbi_layout(title, subtitle)
        )
        
        return {
            'type': 'pie',
            'title': title,
            'position': position,
            'data': json.dumps(fig, cls=PlotlyJSONEncoder),
            'insight': subtitle,
            'chart_type': 'pie'
        }

    def _generate_powerbi_donut_chart(self, df, gemini_info, position) -> Dict:
        """Generate Power BI style donut chart"""
        if not self.categorical_columns:
            return None
        
        cat_col = self.categorical_columns[0]
        value_counts = df[cat_col].value_counts().head(6)
        
        if len(value_counts) < 2:
            return None
        
        colors = self._get_powerbi_colors()
        
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=value_counts.index.astype(str),
            values=value_counts.values,
            hole=0.5,
            marker=dict(
                colors=[colors[i % len(colors)] for i in range(len(value_counts))],
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<br><extra></extra>',
            textposition='outside',
            textinfo='label+percent'
        ))
        
        title = gemini_info.get('title', f"{cat_col.replace('_', ' ').title()} Breakdown") if gemini_info else f"{cat_col.replace('_', ' ').title()} Breakdown"
        subtitle = gemini_info.get('business_insight', f"Central view of {cat_col.replace('_', ' ').lower()} proportions") if gemini_info else None
        
        fig.update_layout(
            **self._get_powerbi_layout(title, subtitle)
        )
        
        return {
            'type': 'donut',
            'title': title,
            'position': position,
            'data': json.dumps(fig, cls=PlotlyJSONEncoder),
            'insight': subtitle,
            'chart_type': 'donut'
        }

    def _generate_powerbi_area_chart(self, df, gemini_info, position) -> Dict:
        """Generate Power BI style area chart"""
        if not (self.datetime_columns and self.numeric_columns):
            return None
        
        dt_col = self.datetime_columns[0]
        num_col = self.numeric_columns[0]
        temp_df = df[[dt_col, num_col]].dropna().sort_values(dt_col)
        
        if temp_df.empty:
            return None
        
        colors = self._get_powerbi_colors()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=temp_df[dt_col],
            y=temp_df[num_col],
            mode='lines',
            fill='tonexty',
            line=dict(color=colors[0], width=3),
            fillcolor=f'rgba({int(colors[0][1:3], 16)}, {int(colors[0][3:5], 16)}, {int(colors[0][5:7], 16)}, 0.3)',
            name=num_col.replace('_', ' ').title(),
            hovertemplate=f'<b>{dt_col.replace("_", " ").title()}</b><br>' +
                         f'{num_col.replace("_", " ").title()}: %{{y:,.2f}}<br>' +
                         '<extra></extra>'
        ))
        
        title = gemini_info.get('title', f"{num_col.replace('_', ' ').title()} Area Chart") if gemini_info else f"{num_col.replace('_', ' ').title()} Area Chart"
        subtitle = gemini_info.get('business_insight', f"Cumulative view of {num_col.replace('_', ' ').lower()} over time") if gemini_info else None
        
        fig.update_layout(
            **self._get_powerbi_layout(title, subtitle),
            xaxis_title=dt_col.replace('_', ' ').title(),
            yaxis_title=num_col.replace('_', ' ').title()
        )
        
        return {
            'type': 'area',
            'title': title,
            'position': position,
            'data': json.dumps(fig, cls=PlotlyJSONEncoder),
            'insight': subtitle,
            'chart_type': 'area'
        }

    def _generate_powerbi_scatter_chart(self, df, gemini_info, position) -> Dict:
        """Generate Power BI style scatter chart"""
        if len(self.numeric_columns) < 2:
            return None
        
        x_col, y_col = self.numeric_columns[:2]
        temp_df = df[[x_col, y_col]].dropna()
        
        if temp_df.empty:
            return None
        
        colors = self._get_powerbi_colors()
        
        # Calculate correlation
        correlation = temp_df[x_col].corr(temp_df[y_col])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=temp_df[x_col],
            y=temp_df[y_col],
            mode='markers',
            marker=dict(
                color=colors[0],
                size=8,
                opacity=0.7,
                line=dict(color='white', width=1)
            ),
            name=f'r = {correlation:.2f}',
            hovertemplate=f'<b>{x_col.replace("_", " ").title()}</b>: %{{x:,.2f}}<br>' +
                         f'<b>{y_col.replace("_", " ").title()}</b>: %{{y:,.2f}}<br>' +
                         '<extra></extra>'
        ))
        
        title = gemini_info.get('title', f"{x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}") if gemini_info else f"{x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}"
        subtitle = gemini_info.get('business_insight', f"Relationship between {x_col.replace('_', ' ').lower()} and {y_col.replace('_', ' ').lower()} (r={correlation:.2f})") if gemini_info else f"Correlation: {correlation:.2f}"
        
        fig.update_layout(
            **self._get_powerbi_layout(title, subtitle),
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title()
        )
        
        return {
            'type': 'scatter',
            'title': title,
            'position': position,
            'data': json.dumps(fig, cls=PlotlyJSONEncoder),
            'insight': subtitle,
            'chart_type': 'scatter'
        }

    def _generate_powerbi_heatmap_chart(self, df, gemini_info, position) -> Dict:
        """Generate Power BI style heatmap chart"""
        if len(self.numeric_columns) < 2:
            return None
        
        corr_matrix = df[self.numeric_columns].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[col.replace('_', ' ').title() for col in corr_matrix.columns],
            y=[col.replace('_', ' ').title() for col in corr_matrix.columns],
            colorscale='RdBu',
            zmid=0,
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<br><extra></extra>',
            colorbar=dict(
                title="Correlation",
                titleside="right",
                tickmode="array",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1", "-0.5", "0", "0.5", "1"]
            )
        ))
        
        title = gemini_info.get('title', "Correlation Heatmap") if gemini_info else "Correlation Heatmap"
        subtitle = gemini_info.get('business_insight', "Shows relationships between numeric variables") if gemini_info else "Shows relationships between numeric variables"
        
        fig.update_layout(
            **self._get_powerbi_layout(title, subtitle),
            margin={'t': 60, 'r': 60, 'b': 80, 'l': 80}
        )
        
        return {
            'type': 'heatmap',
            'title': title,
            'position': position,
            'data': json.dumps(fig, cls=PlotlyJSONEncoder),
            'insight': subtitle,
            'chart_type': 'heatmap'
        }

    def _generate_powerbi_treemap_chart(self, df, gemini_info, position) -> Dict:
        """Generate Power BI style treemap chart"""
        if len(self.categorical_columns) < 2:
            return None
        
        cat1, cat2 = self.categorical_columns[:2]
        treemap_data = df.groupby([cat1, cat2]).size().reset_index(name='count')
        
        # Create hierarchical structure
        parents = []
        labels = []
        values = []
        
        for _, row in treemap_data.iterrows():
            parent = str(row[cat1])
            child = f"{row[cat1]}-{row[cat2]}"
            labels.extend([parent, child])
            parents.extend(['', parent])
            values.extend([0, row['count']])
        
        colors = self._get_powerbi_colors()
        
        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            marker_colorscale='Viridis',
            textinfo="label+value",
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br><extra></extra>'
        ))
        
        title = gemini_info.get('title', f"{cat1.replace('_', ' ').title()} Hierarchy") if gemini_info else f"{cat1.replace('_', ' ').title()} Hierarchy"
        subtitle = gemini_info.get('business_insight', f"Hierarchical view of {cat1.replace('_', ' ').lower()} and {cat2.replace('_', ' ').lower()}") if gemini_info else None
        
        fig.update_layout(
            **self._get_powerbi_layout(title, subtitle)
        )
        
        return {
            'type': 'treemap',
            'title': title,
            'position': position,
            'data': json.dumps(fig, cls=PlotlyJSONEncoder),
            'insight': subtitle,
            'chart_type': 'treemap'
        }

    def _generate_powerbi_gauge_chart(self, df, gemini_info, position) -> Dict:
        """Generate Power BI style gauge chart"""
        if not self.numeric_columns:
            return None
        
        num_col = self.numeric_columns[0]
        data = df[num_col].dropna()
        
        if data.empty:
            return None
        
        mean_val = data.mean()
        max_val = data.max()
        min_val = data.min()
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=mean_val,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{num_col.replace('_', ' ').title()}", 'font': {'size': 16}},
            delta={'reference': data.median(), 'font': {'size': 14}},
            gauge={
                'axis': {'range': [min_val, max_val], 'tickfont': {'size': 12}},
                'bar': {'color': '#1f77b4'},
                'steps': [
                    {'range': [min_val, max_val * 0.5], 'color': '#f0f0f0'},
                    {'range': [max_val * 0.5, max_val * 0.8], 'color': '#d9d9d9'},
                    {'range': [max_val * 0.8, max_val], 'color': '#bfbfbf'}
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': max_val * 0.9
                }
            }
        ))
        
        title = gemini_info.get('title', f"{num_col.replace('_', ' ').title()} Performance") if gemini_info else f"{num_col.replace('_', ' ').title()} Performance"
        subtitle = gemini_info.get('business_insight', f"Current average: {mean_val:.2f}") if gemini_info else f"Current average: {mean_val:.2f}"
        
        fig.update_layout(
            template='plotly_white',
            plot_bgcolor='rgba(255, 255, 255, 1)',
            paper_bgcolor='rgba(255, 255, 255, 1)',
            font={'family': 'Segoe UI, Arial, sans-serif', 'size': 12, 'color': '#323130'},
            margin={'t': 60, 'r': 40, 'b': 60, 'l': 60}
        )
        
        return {
            'type': 'gauge',
            'title': title,
            'position': position,
            'data': json.dumps(fig, cls=PlotlyJSONEncoder),
            'insight': subtitle,
            'chart_type': 'gauge'
        }

    def _generate_generic_powerbi_chart(self, chart_type, df, gemini_info, position) -> Dict:
        """Generate a generic Power BI style chart"""
        # This is a fallback method for chart types not specifically implemented
        return {
            'type': chart_type,
            'title': gemini_info.get('title', f"{chart_type.title()} Chart") if gemini_info else f"{chart_type.title()} Chart",
            'position': position,
            'data': '{}',
            'insight': gemini_info.get('business_insight', f"Chart showing {chart_type} visualization") if gemini_info else None,
            'chart_type': chart_type
        }

    # ML-based scoring system (existing functionality)
    def _score_visualizations(self, df):
        """Advanced ML-based scoring system for visualization recommendations"""
        scores = {}

        # Calculate dataset characteristics for ML scoring
        dataset_features = self._extract_dataset_features(df)
        
        # Time Series Analysis
        if self.datetime_columns and self.numeric_columns:
            time_span = self._calculate_time_span(df)
            data_density = len(df) / max(1, time_span.days if hasattr(time_span, 'days') else 1)
            trend_strength = self._calculate_trend_strength(df)
            
            scores['line'] = min(15, 10 + data_density * 0.1 + trend_strength * 2)
            scores['area'] = min(12, 8 + data_density * 0.08 + trend_strength * 1.5)
        else:
            scores['line'] = 0
            scores['area'] = 0

        # Categorical Analysis
        scores.update(self._score_categorical_visualizations(df))
        scores.update(self._score_numeric_visualizations(df))
        scores.update(self._score_advanced_visualizations(df))
        scores.update(self._score_business_intelligence_charts(df))

        return scores

    def _extract_dataset_features(self, df):
        """Extract features for ML-based scoring"""
        features = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_ratio': len(self.numeric_columns) / len(df.columns),
            'categorical_ratio': len(self.categorical_columns) / len(df.columns),
            'datetime_ratio': len(self.datetime_columns) / len(df.columns),
            'missing_ratio': df.isnull().sum().sum() / (len(df) * len(df.columns)),
            'cardinality_score': np.mean([df[col].nunique() / len(df) for col in df.columns])
        }
        return features

    def _calculate_time_span(self, df):
        """Calculate time span for datetime columns"""
        if not self.datetime_columns:
            return pd.Timedelta(days=1)
        
        dt_col = self.datetime_columns[0]
        dt_data = pd.to_datetime(df[dt_col], errors='coerce').dropna()
        if len(dt_data) < 2:
            return pd.Timedelta(days=1)
        
        return dt_data.max() - dt_data.min()

    def _calculate_trend_strength(self, df):
        """Calculate trend strength for time series data"""
        if not (self.datetime_columns and self.numeric_columns):
            return 0

        dt_col = self.datetime_columns[0]
        num_col = self.numeric_columns[0]

        temp_df = df[[dt_col, num_col]].dropna().sort_values(dt_col)
        if len(temp_df) < 3:
            return 0

        try:
            # Convert datetime to numeric timestamps for correlation
            dt_series = pd.to_datetime(temp_df[dt_col], errors='coerce')
            if dt_series.isna().any():
                # If conversion fails, try to parse with specific format or skip
                return 0

            time_numeric = dt_series.astype('int64') // 10**9  # Convert to seconds since epoch
            correlation = time_numeric.corr(temp_df[num_col])

            return abs(correlation) if not np.isnan(correlation) else 0
        except Exception:
            # If any error occurs in trend calculation, return 0
            return 0

    def _score_categorical_visualizations(self, df):
        """Score categorical visualization types"""
        scores = {}
        scores['bar'] = len(self.categorical_columns) * 6
        
        donut_score = 0
        for cat_col in self.categorical_columns:
            unique_count = df[cat_col].nunique()
            data_balance = 1 - (df[cat_col].value_counts().std() / df[cat_col].value_counts().mean())
            
            if 2 <= unique_count <= 8:
                donut_score += 10 + data_balance * 3
            elif 9 <= unique_count <= 15:
                donut_score += 6 + data_balance * 2
        
        scores['donut'] = donut_score
        scores['pie'] = donut_score * 0.8

        if len(self.categorical_columns) >= 2:
            cross_tab_score = 0
            for cat1, cat2 in zip(self.categorical_columns[:-1], self.categorical_columns[1:]):
                cross_tab = pd.crosstab(df[cat1], df[cat2])
                complexity = min(cross_tab.shape[0] * cross_tab.shape[1], 50)
                cross_tab_score += complexity * 0.3
            
            scores['treemap'] = min(15, cross_tab_score)
        else:
            scores['treemap'] = 0
            
        return scores

    def _score_numeric_visualizations(self, df):
        """Score numeric visualization types"""
        scores = {}
        
        if len(self.numeric_columns) > 1:
            corr_matrix = df[self.numeric_columns].corr()
            strong_correlations = (abs(corr_matrix) > 0.3).sum().sum() - len(self.numeric_columns)
            scores['heatmap'] = min(15, strong_correlations * 2 + len(self.numeric_columns) * 3)
        else:
            scores['heatmap'] = 0

        scatter_score = 0
        if len(self.numeric_columns) > 1:
            corr_matrix = df[self.numeric_columns].corr()
            for i in range(len(self.numeric_columns)):
                for j in range(i+1, len(self.numeric_columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.7:
                        scatter_score += 12
                    elif corr_val > 0.5:
                        scatter_score += 8
                    elif corr_val > 0.3:
                        scatter_score += 5
        
        scores['scatter'] = min(15, scatter_score)
        
        box_score = 0
        for num_col in self.numeric_columns:
            data = df[num_col].dropna()
            if len(data) > 10:
                q1, q3 = data.quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = ((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)).sum()
                outlier_ratio = outliers / len(data)
                
                if outlier_ratio > 0.05:
                    box_score += 10
                elif outlier_ratio > 0.01:
                    box_score += 6
                else:
                    box_score += 3
        
        scores['box'] = min(12, box_score)
        scores['histogram'] = min(12, len(self.numeric_columns) * 4)
        scores['violin'] = min(10, len(self.numeric_columns) * 4)
            
        return scores

    def _score_advanced_visualizations(self, df):
        """Score advanced visualization types"""
        scores = {}
        
        if len(self.numeric_columns) >= 1 and len(self.categorical_columns) >= 1:
            scores['waterfall'] = 7
        
        if len(self.categorical_columns) >= 2:
            scores['sankey'] = 6
        
        if len(self.numeric_columns) >= 3:
            scores['radar'] = 8
            scores['scatter3d'] = 5
        
        return scores

    def _score_business_intelligence_charts(self, df):
        """Score business intelligence specific charts"""
        scores = {}
        scores['gauge'] = 6 if len(self.numeric_columns) > 0 else 0
        
        if len(self.categorical_columns) > 0:
            cat_col = self.categorical_columns[0]
            if 3 <= df[cat_col].nunique() <= 8:
                scores['funnel'] = 8
        
        if len(self.numeric_columns) >= 2:
            scores['bullet'] = 7

        return scores

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
                    fill_value = self.cleaned_df[col].median()
                    self.cleaned_df[col] = self.cleaned_df[col].fillna(fill_value)
                    report['steps_performed'].append(f"Filled {missing_count} missing values in '{col}' with median ({fill_value:.2f})")
                elif col in self.categorical_columns:
                    mode_val = self.cleaned_df[col].mode()
                    fill_value = mode_val[0] if not mode_val.empty else 'Unknown'
                    self.cleaned_df[col] = self.cleaned_df[col].fillna(fill_value)
                    report['steps_performed'].append(f"Filled {missing_count} missing values in '{col}' with mode ('{fill_value}')")
                else:
                    fill_value = 'Unknown'
                    self.cleaned_df[col] = self.cleaned_df[col].fillna(fill_value)
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

        # Safely convert values to float, handling any non-numeric values
        result = {}
        for col in summary.columns:
            col_stats = {}
            for stat, val in summary[col].items():
                try:
                    # Handle different value types
                    if pd.isna(val):
                        col_stats[stat] = None
                    elif isinstance(val, (int, float)):
                        col_stats[stat] = float(val)
                    elif hasattr(val, 'timestamp'):  # Timestamp object
                        col_stats[stat] = float(val.timestamp())
                    else:
                        # Try to convert to string first, then to float if possible
                        str_val = str(val)
                        try:
                            col_stats[stat] = float(str_val)
                        except (ValueError, TypeError):
                            col_stats[stat] = str_val
                except Exception:
                    col_stats[stat] = str(val)
            result[col] = col_stats

        return result

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

    def get_detailed_column_analysis(self):
        """Get detailed analysis for each column"""
        df = self.cleaned_df if self.cleaned_df is not None else self.df
        analysis = {}

        for col in df.columns:
            col_data = df[col].dropna()

            if col in self.numeric_columns:
                # Safely calculate statistics, handling potential Timestamp objects
                try:
                    mean_val = col_data.mean()
                    if hasattr(mean_val, 'timestamp'):  # Timestamp object
                        mean_val = float(mean_val.timestamp())
                    else:
                        mean_val = float(mean_val) if not pd.isna(mean_val) else None
                except:
                    mean_val = None

                try:
                    median_val = col_data.median()
                    if hasattr(median_val, 'timestamp'):  # Timestamp object
                        median_val = float(median_val.timestamp())
                    else:
                        median_val = float(median_val) if not pd.isna(median_val) else None
                except:
                    median_val = None

                try:
                    std_val = col_data.std()
                    std_val = float(std_val) if not pd.isna(std_val) else None
                except:
                    std_val = None

                try:
                    min_val = col_data.min()
                    if hasattr(min_val, 'timestamp'):  # Timestamp object
                        min_val = float(min_val.timestamp())
                    else:
                        min_val = float(min_val) if not pd.isna(min_val) else None
                except:
                    min_val = None

                try:
                    max_val = col_data.max()
                    if hasattr(max_val, 'timestamp'):  # Timestamp object
                        max_val = float(max_val.timestamp())
                    else:
                        max_val = float(max_val) if not pd.isna(max_val) else None
                except:
                    max_val = None

                analysis[col] = {
                    'type': 'numeric',
                    'count': len(col_data),
                    'mean': mean_val,
                    'median': median_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
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

    def generate_smart_kpis(self, df):
        """Generate smart KPIs based on dataset characteristics"""
        kpis = []
        
        # Basic dataset metrics
        info = self.get_enhanced_basic_info()
        
        kpis.append({
            'title': 'Total Records',
            'value': f"{info['shape'][0]:,}",
            'icon': 'database',
            'change': f"{info['shape'][0]:,} rows",
            'change_type': 'neutral'
        })
        
        kpis.append({
            'title': 'Total Columns',
            'value': info['shape'][1],
            'icon': 'columns',
            'change': f"{info['numeric_columns']} numeric, {info['categorical_columns']} categorical",
            'change_type': 'info'
        })
        
        # Data quality metrics
        missing_info = self.analyze_missing_values()
        total_cells = info['shape'][0] * info['shape'][1]
        completeness_score = int(((total_cells - missing_info['total_missing']) / total_cells) * 100) if total_cells > 0 else 100
        
        kpis.append({
            'title': 'Data Completeness',
            'value': f'{completeness_score}%',
            'icon': 'check-circle',
            'change': f"{missing_info['total_missing']} missing values",
            'change_type': 'positive' if completeness_score > 90 else 'warning'
        })
        
        # Numeric analysis KPIs
        if self.numeric_columns:
            numeric_stats = self._calculate_numeric_kpis(df)
            kpis.extend(numeric_stats)
        
        # Categorical analysis KPIs
        if self.categorical_columns:
            categorical_stats = self._calculate_categorical_kpis(df)
            kpis.extend(categorical_stats)
        
        # Time series KPIs
        if self.datetime_columns:
            time_series_stats = self._calculate_time_series_kpis(df)
            kpis.extend(time_series_stats)
        
        # Data diversity metrics
        diversity_metrics = self._calculate_diversity_metrics(df)
        kpis.extend(diversity_metrics)
        
        return kpis

    def _calculate_numeric_kpis(self, df):
        """Calculate KPIs for numeric columns"""
        kpis = []

        for col in self.numeric_columns[:3]:
            # Ensure we're working with numeric data
            try:
                data = pd.to_numeric(df[col], errors='coerce').dropna()
                if data.empty:
                    continue

                mean_val = float(data.mean())
                std_val = float(data.std())

                # Handle division by zero and invalid values
                if mean_val != 0 and not pd.isna(mean_val) and not pd.isna(std_val):
                    cv = (std_val / mean_val * 100)
                else:
                    cv = 0

                kpis.append({
                    'title': f'{col.replace("_", " ").title()} Average',
                    'value': f'{mean_val:.1f}',
                    'icon': 'calculator',
                    'change': f'Std: {std_val:.1f}',
                    'change_type': 'info'
                })

                if cv > 0:
                    consistency_level = 'High' if cv < 20 else 'Medium' if cv < 50 else 'Low'
                    kpis.append({
                        'title': f'{col.replace("_", " ").title()} Consistency',
                        'value': consistency_level,
                        'icon': 'chart-line',
                        'change': f'CV: {cv:.1f}%',
                        'change_type': 'positive' if cv < 20 else 'warning' if cv < 50 else 'negative'
                    })
            except Exception:
                # Skip this column if we can't calculate numeric KPIs
                continue

        return kpis

    def _calculate_categorical_kpis(self, df):
        """Calculate KPIs for categorical columns"""
        kpis = []
        
        for col in self.categorical_columns[:3]:
            data = df[col].dropna()
            if data.empty:
                continue
                
            unique_count = data.nunique()
            total_count = len(data)
            diversity_ratio = unique_count / total_count
            
            most_frequent = data.value_counts().iloc[0]
            frequency_percentage = (most_frequent / total_count) * 100
            
            kpis.append({
                'title': f'{col.replace("_", " ").title()} Categories',
                'value': unique_count,
                'icon': 'tags',
                'change': f'{frequency_percentage:.1f}% most frequent',
                'change_type': 'info'
            })
            
            diversity_level = 'High' if diversity_ratio > 0.8 else 'Medium' if diversity_ratio > 0.5 else 'Low'
            kpis.append({
                'title': f'{col.replace("_", " ").title()} Diversity',
                'value': diversity_level,
                'icon': 'chart-pie',
                'change': f'{diversity_ratio:.2f} ratio',
                'change_type': 'positive' if diversity_ratio > 0.8 else 'warning' if diversity_ratio > 0.5 else 'negative'
            })
        
        return kpis

    def _calculate_time_series_kpis(self, df):
        """Calculate KPIs for time series data"""
        kpis = []

        if not (self.datetime_columns and self.numeric_columns):
            return kpis

        dt_col = self.datetime_columns[0]
        num_col = self.numeric_columns[0]

        try:
            # Ensure datetime column is properly parsed
            temp_df = df[[dt_col, num_col]].copy()
            temp_df[dt_col] = pd.to_datetime(temp_df[dt_col], errors='coerce')
            temp_df = temp_df.dropna().sort_values(dt_col)

            if len(temp_df) < 2:
                return kpis

            # Calculate time span safely
            time_span = temp_df[dt_col].max() - temp_df[dt_col].min()
            if hasattr(time_span, 'days'):
                time_span_str = f"{time_span.days} days"
            else:
                time_span_str = str(time_span)

            kpis.append({
                'title': 'Time Span',
                'value': time_span_str,
                'icon': 'calendar',
                'change': f'{len(temp_df)} data points',
                'change_type': 'info'
            })

            if len(temp_df) >= 3:
                # Ensure numeric column is numeric
                temp_df[num_col] = pd.to_numeric(temp_df[num_col], errors='coerce')
                temp_df = temp_df.dropna()

                if len(temp_df) >= 3:
                    first_half = temp_df.iloc[:len(temp_df)//2][num_col].mean()
                    second_half = temp_df.iloc[len(temp_df)//2:][num_col].mean()

                    if not pd.isna(first_half) and not pd.isna(second_half):
                        trend_direction = 'Increasing' if second_half > first_half else 'Decreasing'
                        trend_strength = abs((second_half - first_half) / first_half * 100) if first_half != 0 else 0

                        kpis.append({
                            'title': f'{num_col.replace("_", " ").title()} Trend',
                            'value': trend_direction,
                            'icon': 'trending-up' if second_half > first_half else 'trending-down',
                            'change': f'{trend_strength:.1f}% change',
                            'change_type': 'positive' if second_half > first_half else 'negative'
                        })
        except Exception:
            # Skip time series KPIs if calculation fails
            pass

        return kpis

    def _calculate_diversity_metrics(self, df):
        """Calculate data diversity and uniqueness metrics"""
        kpis = []
        
        avg_unique = np.mean([df[col].nunique() for col in df.columns])
        kpis.append({
            'title': 'Avg Unique Values',
            'value': f"{avg_unique:.0f}",
            'icon': 'chart-bar',
            'change': 'Per column average',
            'change_type': 'info'
        })
        
        total_cells = len(df) * len(df.columns)
        non_null_cells = total_cells - df.isnull().sum().sum()
        density = (non_null_cells / total_cells) * 100 if total_cells > 0 else 100
        
        kpis.append({
            'title': 'Data Density',
            'value': f'{density:.1f}%',
            'icon': 'database',
            'change': f'{non_null_cells:,} / {total_cells:,} cells',
            'change_type': 'positive' if density > 90 else 'warning' if density > 75 else 'negative'
        })
        
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        kpis.append({
            'title': 'Memory Usage',
            'value': f'{memory_mb:.1f} MB',
            'icon': 'memory',
            'change': f'{len(df):,} rows',
            'change_type': 'info'
        })
        
        return kpis

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
