import json
import pandas as pd
import requests
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import re
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Structure for query results"""
    query_type: str
    data: List[Dict]
    chart_type: str
    title: str
    insight: str
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    sql_query: Optional[str] = None
    confidence: float = 0.0

class LLMQueryProcessor:
    """Handles LLM-based query processing using OpenRouter"""
    
    def __init__(self, api_key: str, model: str, base_url: str):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        
    def process_query(self, query: str, df: pd.DataFrame, dataset_info: Dict) -> QueryResult:
        """Process natural language query using LLM"""
        try:
            # Ensure df columns are stripped of whitespace for consistency
            df.columns = df.columns.astype(str).str.strip()

            # Create context about the dataset
            context = self._create_dataset_context(df, dataset_info)
            
            # Generate system prompt
            system_prompt = self._create_system_prompt(context)
            
            # Create user prompt
            user_prompt = f"""
            User Query: {query}
            
            Please analyze this query and provide a structured response with:
            1. The type of analysis needed
            2. The appropriate chart type
            3. The data processing steps
            4. An insight about the results
            
            Return your response in the following JSON format. Ensure all column names in 'columns_needed', 'group_by', 'sort_by', and 'filter_conditions' exactly match the dataset's columns (case-sensitive) as provided in the Dataset Context.
            
            For 'aggregation' or 'comparison' queries, if a grouping dimension is implied (e.g., "by Product", "per Region"), ensure 'group_by' is set to the correct column name and 'columns_needed' contains the metric to aggregate.
            
            {{
                "query_type": "aggregation|comparison|trend|summary|filter",
                "chart_type": "bar|line|pie|scatter|table",
                "columns_needed": ["column1", "column2"],
                "aggregation": "sum|count|avg|min|max",
                "group_by": "column_name",
                "filter_conditions": {{}},
                "sort_by": "column_name",
                "sort_order": "asc|desc",
                "limit": 10,
                "title": "Chart Title",
                "insight_template": "template for insight"
            }}
            """
            
            # Call LLM
            response_content = self._call_llm(system_prompt, user_prompt)
            logger.info(f"Raw LLM response content: {response_content}") # DEBUG: Log raw LLM response
            
            # Parse LLM response
            parsed_response = self._parse_llm_response(response_content)
            logger.info(f"Parsed LLM response: {parsed_response}") # DEBUG: Log parsed LLM response
            
            # Execute the query on the dataframe
            result = self._execute_query(df, parsed_response, dataset_info)
            
            return result
            
        except Exception as e:
            logger.error(f"LLM query processing error: {e}", exc_info=True) # Log full traceback
            # Fallback to rule-based processing
            return self._fallback_processing(query, df, dataset_info)
        
    def _create_dataset_context(self, df: pd.DataFrame, dataset_info: Dict) -> str:
        """Create context about the dataset for the LLM"""
        context_parts = []
        
        # Basic info
        context_parts.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns.")
        
        # Column information
        context_parts.append("Columns and their types:")
        for col in df.columns: # Iterate directly over df.columns for actual names
            dtype = dataset_info['data_types'].get(col, 'unknown') # Get dtype from dataset_info
            
            sample_values = []
            if col in df.columns: # Ensure column exists before trying to get samples
                try:
                    # Handle potential non-string values in sample_values for text columns
                    if dtype == 'text':
                        sample_values = [str(x) for x in df[col].dropna().head(3).tolist()]
                    else:
                        sample_values = df[col].dropna().head(3).tolist()
                except Exception as e:
                    logger.warning(f"Could not get sample values for column '{col}': {e}")
                    sample_values = ["(error getting samples)"]
            
            context_parts.append(f"- '{col}' (Type: {dtype}): Sample values: {sample_values}")
        
        # Numeric column stats
        numeric_cols = [col for col, dtype in dataset_info['data_types'].items() if dtype == 'numeric']
        if numeric_cols:
            context_parts.append(f"Numeric columns: {', '.join(numeric_cols)}")
        
        # Text/categorical columns
        text_cols = [col for col, dtype in dataset_info['data_types'].items() if dtype == 'text']
        if text_cols:
            context_parts.append(f"Categorical columns: {', '.join(text_cols)}")
        
        # Date columns
        date_cols = [col for col, dtype in dataset_info['data_types'].items() if dtype == 'date']
        if date_cols:
            context_parts.append(f"Date columns: {', '.join(date_cols)}")
        
        return "\n".join(context_parts)
    
    def _create_system_prompt(self, context: str) -> str:
        """Create system prompt for the LLM"""
        return f"""
        You are a data analyst AI that helps users analyze their datasets through natural language queries.
        
        Dataset Context:
        {context}
        
        Your task is to:
        1. Understand the user's query intent
        2. Identify the appropriate analysis type
        3. Determine the best visualization
        4. Provide specific instructions for data processing
        
        Always respond with valid JSON that can be parsed programmatically.
        Ensure column names in your JSON response exactly match the column names provided in the Dataset Context, including case.
        If a column is not found, suggest a relevant existing column or indicate it's not possible.
        For aggregation or comparison queries, if a grouping dimension is implied (e.g., "by Product", "per Region"), ensure 'group_by' is set to the correct column name.
        For trend analysis, ensure the 'group_by' column is a date type and 'columns_needed' are numeric.
        """
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        full_url = f"{self.base_url}chat/completions"

        try:
            response = requests.post(full_url, headers=headers, json=payload)
            
            logger.info(f"OpenRouter API Response Status Code: {response.status_code}")
            logger.info(f"OpenRouter API Raw Response Text: {response.text[:500]}...")

            response.raise_for_status()
            
            response_json = response.json()
            
            if response_json and 'choices' in response_json and len(response_json['choices']) > 0:
                return response_json['choices'][0]['message']['content']
            else:
                raise ValueError(f"Unexpected OpenRouter API response structure: {response_json}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API call failed: {e}")
            raise
        except ValueError as e:
            logger.error(f"Error parsing OpenRouter response: {e}")
            raise
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response into structured format"""
        try:
            json_match = re.search(r'```json\n(.*)\n```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No valid JSON found in LLM response")
            
            parsed_json = json.loads(json_str)

            if 'columns_needed' in parsed_json and isinstance(parsed_json['columns_needed'], list):
                parsed_json['columns_needed'] = [col.strip("'\"") for col in parsed_json['columns_needed']]
            if 'group_by' in parsed_json and isinstance(parsed_json['group_by'], str):
                parsed_json['group_by'] = parsed_json['group_by'].strip("'\"")
            if 'sort_by' in parsed_json and isinstance(parsed_json['sort_by'], str):
                parsed_json['sort_by'] = parsed_json['sort_by'].strip("'\"")
            
            return parsed_json
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error in LLM response: {e}. Raw response: {response}")
            return {
                "query_type": "summary",
                "chart_type": "table",
                "columns_needed": [],
                "aggregation": "count",
                "title": "Data Analysis (Fallback)",
                "insight_template": "Analysis completed (fallback due to parsing error)"
            }
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}. Raw response: {response}")
            return {
                "query_type": "summary",
                "chart_type": "table",
                "columns_needed": [],
                "aggregation": "count",
                "title": "Data Analysis (Fallback)",
                "insight_template": "Analysis completed (fallback due to parsing error)"
            }
    
    def _execute_query(self, df: pd.DataFrame, query_config: Dict, dataset_info: Dict) -> QueryResult:
        """Execute the query based on LLM instructions"""
        try:
            query_type = query_config.get('query_type', 'summary')
            chart_type = query_config.get('chart_type', 'table')
            
            df.columns = df.columns.astype(str).str.strip()
            df_cols_lower = {col.lower(): col for col in df.columns}

            def get_actual_col_name(col_name_from_llm):
                if col_name_from_llm in df.columns:
                    return col_name_from_llm
                return df_cols_lower.get(col_name_from_llm.lower())

            actual_columns_needed = []
            if 'columns_needed' in query_config and query_config['columns_needed']:
                for col in query_config['columns_needed']:
                    actual_col = get_actual_col_name(col)
                    if actual_col:
                        actual_columns_needed.append(actual_col)
                    else:
                        raise ValueError(f"Column '{col}' requested by LLM not found in dataset. Available: {list(df.columns)}")
            query_config['columns_needed'] = actual_columns_needed

            actual_group_by_col = None
            if 'group_by' in query_config and query_config['group_by']:
                actual_group_by_col = get_actual_col_name(query_config['group_by'])
                if not actual_group_by_col:
                    raise ValueError(f"Group-by column '{query_config['group_by']}' requested by LLM not found in dataset. Available: {list(df.columns)}")
            query_config['group_by'] = actual_group_by_col

            actual_sort_by_col = None
            if 'sort_by' in query_config and query_config['sort_by']:
                actual_sort_by_col = get_actual_col_name(query_config['sort_by'])
            query_config['sort_by'] = actual_sort_by_col

            # --- NEW LOGIC FOR FILTERING FIRST ---
            processed_df = df.copy()
            filter_conditions = query_config.get('filter_conditions', {})
            if filter_conditions:
                try:
                    processed_df = self._apply_filters_to_df(processed_df, filter_conditions, dataset_info)
                    if processed_df.empty:
                        return self._create_error_result("No data found after applying filters. Please adjust your query or filters.")
                except Exception as e:
                    return self._create_error_result(f"Error applying filters: {e}")
            # --- END NEW LOGIC ---

            if query_type == 'aggregation':
                return self._handle_aggregation(processed_df, query_config, dataset_info)
            elif query_type == 'comparison':
                return self._handle_comparison(processed_df, query_config, dataset_info)
            elif query_type == 'trend':
                return self._handle_trend(processed_df, query_config, dataset_info)
            elif query_type == 'filter': # If query_type is explicitly 'filter', just return the filtered data as a table
                return self._handle_summary(processed_df, query_config, dataset_info) # Summary of filtered data
            else:
                return self._handle_summary(processed_df, query_config, dataset_info)
                
        except ValueError as ve:
            logger.error(f"Query execution validation error: {ve}")
            return self._create_error_result(f"Data processing error: {ve}")
        except KeyError as ke:
            logger.error(f"Query execution KeyError: {ke}. This often means a column name from LLM was incorrect.")
            return self._create_error_result(f"Data processing error: Missing column '{ke}'. Please check your query or dataset columns.")
        except Exception as e:
            logger.error(f"Query execution general error: {e}", exc_info=True)
            return self._create_error_result(f"An unexpected error occurred during query execution: {e}")
    
    def _apply_filters_to_df(self, df_to_filter: pd.DataFrame, filter_conditions: Dict, dataset_info: Dict) -> pd.DataFrame:
        """Applies filter conditions to a DataFrame and returns the filtered DataFrame."""
        filtered_df = df_to_filter.copy()
        
        for col, condition in filter_conditions.items():
            actual_col = next((c for c in filtered_df.columns if c.lower() == col.lower()), None)
            if not actual_col:
                raise ValueError(f"Filter column '{col}' not found in dataset.")
            
            try:
                if dataset_info['data_types'].get(actual_col) == 'numeric':
                    # Ensure both column and condition are numeric for comparison
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[actual_col], errors='coerce') == pd.to_numeric(condition, errors='coerce')]
                else:
                    # For text/date, direct comparison
                    filtered_df = filtered_df[filtered_df[actual_col] == condition]
            except Exception as e:
                logger.warning(f"Failed to apply filter on column {actual_col} with condition {condition}: {e}")
                raise ValueError(f"Filter error on column '{actual_col}': {e}")
        return filtered_df

    def _handle_aggregation(self, df: pd.DataFrame, config: Dict, dataset_info: Dict) -> QueryResult:
        """Handle aggregation queries"""
        group_by = config.get('group_by')
        aggregation = config.get('aggregation', 'sum')
        columns_needed = config.get('columns_needed', [])
        
        # --- NEW LOGIC START (Inference if LLM misses group_by for aggregation/comparison) ---
        if not group_by and (config.get('query_type') == 'aggregation' or config.get('query_type') == 'comparison'):
            text_cols = [col for col, dtype in dataset_info['data_types'].items() if dtype == 'text']
            for col_needed in columns_needed:
                if col_needed in text_cols: # If a metric is also a text column, it's likely the group_by
                    group_by = col_needed
                    logger.info(f"Inferred group_by column: {group_by} from columns_needed (text type).")
                    break
            if not group_by and text_cols: # If still no group_by, take the first text column
                group_by = text_cols[0]
                logger.info(f"Inferred group_by column: {group_by} from dataset text columns.")
            
            if group_by:
                config['group_by'] = group_by # Update config for consistency
            else:
                return self._create_error_result("Missing required parameters for aggregation (group_by or columns_needed). Could not infer a suitable group_by column.")
        # --- NEW LOGIC END ---

        if not group_by or not columns_needed:
            return self._create_error_result("Missing required parameters for aggregation (group_by or columns_needed).")
        
        if group_by not in df.columns:
            return self._create_error_result(f"Group-by column '{group_by}' not found in DataFrame.")
        
        for col in columns_needed:
            if col not in df.columns:
                return self._create_error_result(f"Value column '{col}' not found in DataFrame for aggregation.")
            if dataset_info['data_types'].get(col) != 'numeric':
                return self._create_error_result(f"Column '{col}' is not numeric for aggregation. Its type is {dataset_info['data_types'].get(col)}.")
        
        try:
            if aggregation == 'sum':
                result = df.groupby(group_by)[columns_needed].sum(numeric_only=True).reset_index()
            elif aggregation == 'count':
                result = df.groupby(group_by)[columns_needed].count().reset_index()
            elif aggregation == 'avg':
                result = df.groupby(group_by)[columns_needed].mean(numeric_only=True).reset_index()
            elif aggregation == 'min':
                result = df.groupby(group_by)[columns_needed].min(numeric_only=True).reset_index()
            elif aggregation == 'max':
                result = df.groupby(group_by)[columns_needed].max(numeric_only=True).reset_index()
            else:
                result = df.groupby(group_by)[columns_needed].sum(numeric_only=True).reset_index()
        except Exception as e:
            return self._create_error_result(f"Error during aggregation calculation: {e}")

        sort_by_col = config.get('sort_by')
        if sort_by_col and sort_by_col in result.columns:
            ascending = config.get('sort_order', 'desc') == 'asc'
            result = result.sort_values(sort_by_col, ascending=ascending)
        elif columns_needed and columns_needed[0] in result.columns:
            ascending = config.get('sort_order', 'desc') == 'asc'
            result = result.sort_values(columns_needed[0], ascending=ascending)
        
        if config.get('limit'):
            result = result.head(config['limit'])
        
        data = result.to_dict('records')
        for row in data:
            for key, value in row.items():
                if isinstance(value, (np.float32, np.float64)):
                    row[key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    row[key] = int(value)
                elif isinstance(value, pd.Period):
                    row[key] = str(value)
        
        insight = self._generate_insight(data, config, aggregation, group_by, columns_needed)
        
        return QueryResult(
            query_type='aggregation',
            data=data,
            chart_type=config.get('chart_type', 'bar'),
            title=config.get('title', f'{aggregation.title()} by {group_by}'),
            insight=insight,
            x_axis=group_by,
            y_axis=columns_needed[0] if columns_needed else None,
            confidence=0.8
        )
    
    def _handle_comparison(self, df: pd.DataFrame, config: Dict, dataset_info: Dict) -> QueryResult:
        """Handle comparison queries (delegates to aggregation)"""
        return self._handle_aggregation(df, config, dataset_info)
    
    def _handle_trend(self, df: pd.DataFrame, config: Dict, dataset_info: Dict) -> QueryResult:
        """Handle trend analysis queries"""
        date_col = config.get('group_by')
        value_cols = config.get('columns_needed', [])
        
        if not date_col or not value_cols:
            date_cols_info = [col for col, dtype in dataset_info['data_types'].items() if dtype == 'date']
            if not date_cols_info:
                date_cols_info = [col for col in df.columns if any(keyword in col.lower() 
                                    for keyword in ['date', 'time', 'month', 'year'])]
            if not date_cols_info:
                return self._create_error_result("No suitable date column found for trend analysis.")
            date_col = date_cols_info[0]

            if not value_cols:
                numeric_cols = [col for col, dtype in dataset_info['data_types'].items() if dtype == 'numeric']
                if not numeric_cols:
                    return self._create_error_result("No numeric columns found for trend analysis.")
                value_cols = [numeric_cols[0]]

        if date_col not in df.columns or value_cols[0] not in df.columns:
            return self._create_error_result(f"Date column '{date_col}' or value column '{value_cols[0]}' not found for trend analysis.")
        
        if dataset_info['data_types'].get(value_cols[0]) != 'numeric':
            return self._create_error_result(f"Value column '{value_cols[0]}' is not numeric for trend analysis. Its type is {dataset_info['data_types'].get(value_cols[0])}.")

        temp_df = df.copy()
        temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
        temp_df = temp_df.dropna(subset=[date_col])

        if temp_df.empty:
            return self._create_error_result("No valid date entries after parsing for trend analysis.")

        min_date = temp_df[date_col].min()
        max_date = temp_df[date_col].max()
        
        if pd.isna(min_date) or pd.isna(max_date):
             return self._create_error_result("Invalid date range for trend analysis (min/max date is NaN).")

        date_range_days = (max_date - min_date).days

        if date_range_days > 365 * 2:
            freq = 'Y'
        elif date_range_days > 60:
            freq = 'M'
        else:
            freq = 'D'

        try:
            result = temp_df.groupby(temp_df[date_col].dt.to_period(freq))[value_cols].sum(numeric_only=True).reset_index()
            result[date_col] = result[date_col].astype(str)
            result = result.sort_values(date_col)
            data = result.to_dict('records')
            for row in data:
                for key, value in row.items():
                    if isinstance(value, (np.float32, np.float64)):
                        row[key] = float(value)
                    elif isinstance(value, (np.int32, np.int64)):
                        row[key] = int(value)
                    elif isinstance(value, pd.Period):
                        row[key] = str(value)
        except Exception as e:
            return self._create_error_result(f"Error during trend aggregation calculation: {e}")
        
        insight = f"Trend analysis showing {value_cols[0]} over {date_col} (grouped by {freq})"
        
        return QueryResult(
            query_type='trend',
            data=data,
            chart_type='line',
            title=config.get('title', f'{value_cols[0]} Trend Over Time'),
            insight=insight,
            x_axis=date_col,
            y_axis=value_cols[0],
            confidence=0.8
        )
    
    def _handle_filter(self, df: pd.DataFrame, config: Dict, dataset_info: Dict) -> QueryResult:
        """Handle filter queries - this function is now primarily for when query_type is explicitly 'filter' """
        # The actual filtering logic has been moved to _apply_filters_to_df
        # This function will now just return a summary of the filtered data if query_type is 'filter'
        return self._handle_summary(df, config, dataset_info) # df here is already filtered by _execute_query
    
    def _handle_summary(self, df: pd.DataFrame, config: Dict, dataset_info: Dict) -> QueryResult:
        """Handle summary queries"""
        numeric_cols = [col for col, dtype in dataset_info['data_types'].items() if dtype == 'numeric']
        
        summary_data = []
        for col in numeric_cols:
            if col in df.columns and not df[col].isna().all():
                summary_data.append({
                    'metric': col,
                    'total': float(df[col].sum()),
                    'average': float(df[col].mean()),
                    'count': int(df[col].count())
                })
        
        return QueryResult(
            query_type='summary',
            data=summary_data,
            chart_type='table',
            title='Data Summary',
            insight=f"Dataset contains {len(df)} records and {len(numeric_cols)} numeric columns",
            confidence=0.9
        )
    
    def _generate_insight(self, data: List[Dict], config: Dict, aggregation: str, group_by: str, columns_needed: List[str]) -> str:
        """Generate insight from the data"""
        if not data:
            return "No data available for insight generation."
        
        insight_template = config.get('insight_template')
        if insight_template:
            # Attempt to replace placeholders like [sum of Gross Sales]
            # This requires a more robust replacement logic
            try:
                # First, try direct format with data[0]
                formatted_insight = insight_template.format(**data[0])
            except (KeyError, IndexError):
                formatted_insight = insight_template # Fallback to original if direct format fails

            # Now, try to replace specific patterns like [sum of Column]
            for col_name in columns_needed:
                if f"[sum of {col_name}]" in formatted_insight and col_name in data[0]:
                    formatted_value = f"{data[0][col_name]:,.0f}"
                    if 'sales' in col_name.lower() or 'revenue' in col_name.lower() or 'profit' in col_name.lower():
                        formatted_value = f"${formatted_value}"
                    formatted_insight = formatted_insight.replace(f"[sum of {col_name}]", formatted_value)
                # Add other aggregations if needed (e.g., [avg of Column])
            
            return formatted_insight

        top_record = data[0]
        value_col = columns_needed[0] if columns_needed else None
        
        if value_col and group_by and group_by in top_record and value_col in top_record:
            value = top_record.get(value_col, 0)
            category = top_record.get(group_by, 'Unknown')
            
            if 'sales' in value_col.lower() or 'revenue' in value_col.lower() or 'profit' in value_col.lower():
                formatted_value = f"${value:,.0f}"
            else:
                formatted_value = f"{value:,.0f}"

            return f"Top {group_by} is '{category}' with {aggregation} of {formatted_value}."
        
        return f"Analysis complete with {len(data)} results."
    
    def _create_error_result(self, error_message: str) -> QueryResult:
        """Create error result"""
        logger.error(f"Creating error result: {error_message}")
        return QueryResult(
            query_type='error',
            data=[],
            chart_type='table',
            title='Error',
            insight=f"Error: {error_message}",
            confidence=0.0
        )
    
    def _fallback_processing(self, query: str, df: pd.DataFrame, dataset_info: Dict) -> QueryResult:
        """Fallback to rule-based processing when LLM fails"""
        logger.info("Using fallback rule-based processing")
        
        query_lower = query.lower()
        
        if 'top' in query_lower or 'best' in query_lower or 'highest' in query_lower:
            return self._create_top_analysis(df, dataset_info)
        elif 'trend' in query_lower or 'time' in query_lower or 'monthly' in query_lower or 'yearly' in query_lower:
            return self._create_trend_analysis_fallback(df, dataset_info)
        elif 'summary' in query_lower or 'overview' in query_lower or 'statistics' in query_lower:
            return self._create_summary_analysis_fallback(df, dataset_info)
        else:
            return self._create_summary_analysis_fallback(df, dataset_info)
    
    def _create_top_analysis(self, df: pd.DataFrame, dataset_info: Dict) -> QueryResult:
        """Create top analysis using fallback logic"""
        text_cols = [col for col, dtype in dataset_info['data_types'].items() if dtype == 'text']
        numeric_cols = [col for col, dtype in dataset_info['data_types'].items() if dtype == 'numeric']
        
        if not text_cols or not numeric_cols:
            return self._create_error_result("Fallback: Unable to identify grouping and value columns for 'top' analysis.")
        
        group_col = text_cols[0]
        value_col = numeric_cols[0]
        
        if group_col not in df.columns or value_col not in df.columns:
            return self._create_error_result(f"Fallback: Columns '{group_col}' or '{value_col}' not found for 'top' analysis.")

        try:
            result = df.groupby(group_col)[value_col].sum(numeric_only=True).sort_values(ascending=False).head(10)
            data = [{'name': str(k), 'value': float(v)} for k, v in result.items()]
        except Exception as e:
            return self._create_error_result(f"Fallback: Error during 'top' analysis aggregation: {e}")
        
        return QueryResult(
            query_type='comparison',
            data=data,
            chart_type='bar',
            title=f'Top {group_col} by {value_col}',
            insight=f"Top {group_col} is {data[0]['name']} with {data[0]['value']:,.0f}" if data else "No data",
            x_axis='name',
            y_axis='value',
            confidence=0.6
        )
    
    def _create_trend_analysis_fallback(self, df: pd.DataFrame, dataset_info: Dict) -> QueryResult:
        """Create trend analysis using fallback logic"""
        date_cols = [col for col, dtype in dataset_info['data_types'].items() if dtype == 'date']
        if not date_cols:
            date_cols = [col for col in df.columns if any(keyword in col.lower() 
                                for keyword in ['date', 'time', 'month', 'year'])]
        
        numeric_cols = [col for col, dtype in dataset_info['data_types'].items() if dtype == 'numeric']
        
        if not date_cols or not numeric_cols:
            return self._create_error_result("Fallback: Unable to find date and numeric columns for trend analysis.")
        
        date_col = date_cols[0]
        value_col = numeric_cols[0]
        
        if date_col not in df.columns or value_col not in df.columns:
            return self._create_error_result(f"Fallback: Columns '{date_col}' or '{value_col}' not found for trend analysis.")

        temp_df = df.copy()
        temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
        temp_df = temp_df.dropna(subset=[date_col])

        if temp_df.empty:
            return self._create_error_result("Fallback: No valid date entries after parsing for trend analysis.")

        min_date = temp_df[date_col].min()
        max_date = temp_df[date_col].max()
        
        if pd.isna(min_date) or pd.isna(max_date):
            return self._create_error_result("Fallback: Invalid date range for trend analysis.")

        date_range_days = (max_date - min_date).days

        if date_range_days > 365 * 2:
            freq = 'Y'
        elif date_range_days > 60:
            freq = 'M'
        else:
            freq = 'D'

        try:
            result = temp_df.groupby(temp_df[date_col].dt.to_period(freq))[value_col].sum(numeric_only=True).reset_index()
            result[date_col] = result[date_col].astype(str)
            result = result.sort_values(date_col)
            data = result.to_dict('records')
            for row in data:
                for key, value in row.items():
                    if isinstance(value, (np.float32, np.float64)):
                        row[key] = float(value)
                    elif isinstance(value, (np.int32, np.int64)):
                        row[key] = int(value)
        except Exception as e:
            return self._create_error_result(f"Fallback: Error during trend analysis: {e}")
        
        return QueryResult(
            query_type='trend',
            data=data,
            chart_type='line',
            title=f'{value_col} Trend Over Time (Fallback)',
            insight=f"Trend analysis showing {value_col} over {date_col} (grouped by {freq})",
            x_axis=date_col,
            y_axis=value_col,
            confidence=0.6
        )
    
    def _create_summary_analysis_fallback(self, df: pd.DataFrame, dataset_info: Dict) -> QueryResult:
        """Create summary analysis using fallback logic"""
        numeric_cols = [col for col, dtype in dataset_info['data_types'].items() if dtype == 'numeric']
        
        summary_data = []
        for col in numeric_cols:
            if col in df.columns and not df[col].isna().all():
                summary_data.append({
                    'metric': col,
                    'total': float(df[col].sum()),
                    'average': float(df[col].mean()),
                    'count': int(df[col].count())
                })
        
        return QueryResult(
            query_type='summary',
            data=summary_data,
            chart_type='table',
            title='Data Summary (Local Fallback)',
            insight=f"Summary of {len(df)} records with {len(numeric_cols)} metrics",
            confidence=0.7
        )
