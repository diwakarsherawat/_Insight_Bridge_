const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:5000';
import React, { useState, useRef, useEffect } from 'react';
import { Upload, FileText, BarChart3, Download, Sparkles, AlertCircle, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';

const NLPBusinessIntelligence = () => {
  // Cloudinary Configuration (from your Cloudinary Dashboard)
  const CLOUDINARY_CLOUD_NAME = "dxkgcikx1"; // <--- REPLACE THIS
  const CLOUDINARY_UPLOAD_PRESET = "nlp_bi_upload"; // <--- REPLACE THIS with the preset name you created

  // Simulate a user ID for local development (no Firebase Auth needed for this setup)
  const [userId, setUserId] = useState("your_unique_user_id"); // You can change this to any string

  // App-specific states
  const [uploadedFile, setUploadedFile] = useState(null);
  const [data, setData] = useState([]); // Raw data from uploaded file (for preview)
  const [columns, setColumns] = useState([]); // Columns of the uploaded data
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null); // Results from NLP query
  const [isLoading, setIsLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(''); // 'processing', 'success', 'error'
  const [errorMessage, setErrorMessage] = useState(''); // For displaying errors
  const fileInputRef = useRef(null);

  // State to hold the dataset ID received from the backend after upload
  const [currentDatasetId, setCurrentDatasetId] = useState(null);

  // Suggested queries for user guidance
  const suggestedQueries = [
    "Show me the top products by sales",
    "What's the sales breakdown by region?",
    "Display the monthly sales trend",
    "Give me a summary of the data"
  ];

  // Function to handle file upload to Cloudinary and then send metadata to Flask
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploadedFile(file);
    setUploadStatus('processing');
    setErrorMessage(''); // Clear previous errors
    setResults(null); // Clear previous results

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('upload_preset', CLOUDINARY_UPLOAD_PRESET);
      formData.append('folder', `nlp-bi-app/${userId}`); // Optional: organize files by user ID

      // 1. Upload file directly to Cloudinary
      const cloudinaryResponse = await fetch(
        `https://api.cloudinary.com/v1_1/${CLOUDINARY_CLOUD_NAME}/auto/upload`,
        {
          method: 'POST',
          body: formData,
        }
      );

      const cloudinaryResult = await cloudinaryResponse.json();

      if (!cloudinaryResponse.ok || cloudinaryResult.error) {
        throw new Error(cloudinaryResult.error?.message || 'Cloudinary upload failed.');
      }

      const fileDownloadURL = cloudinaryResult.secure_url; // Get the secure URL from Cloudinary
      const datasetId = uuidv4(); // Generate a unique ID for the dataset

      console.log('File uploaded to Cloudinary:', fileDownloadURL);

      // 2. Send file metadata (including Cloudinary URL) to Flask backend for analysis and MongoDB storage
      const fileAnalysisResponse = await fetch('${BACKEND_URL}/api/analyze_file_metadata', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_storage_url: fileDownloadURL,
          filename: file.name,
          file_extension: file.name.split('.').pop(),
          dataset_id: datasetId,
          userId: userId, // Pass userId for backend to store in MongoDB
        }),
      });

      const analysisResult = await fileAnalysisResponse.json();

      if (!fileAnalysisResponse.ok) {
        throw new Error(analysisResult.error || 'File analysis failed on backend.');
      }

      setUploadStatus('success');
      setData(analysisResult.sample_data);
      setColumns(analysisResult.columns);
      setCurrentDatasetId(datasetId); // Set the current dataset ID
      setErrorMessage(''); // Clear any previous errors

    } catch (error) {
      setUploadStatus('error');
      setErrorMessage(`Upload or analysis failed: ${error.message}`);
      console.error('Upload/Analysis error:', error);
    }
  };

  // Function to process NLP query using the Flask backend
  const processNLPQuery = async (userQuery) => {
    if (!currentDatasetId) {
      setErrorMessage('Please upload a file first.');
      return;
    }
    setIsLoading(true);
    setErrorMessage(''); // Clear previous errors
    setResults(null); // Clear previous results

    try {
      const response = await fetch('${BACKEND_URL}/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userQuery,
          dataset_id: currentDatasetId,
          userId: userId, // Pass userId for backend to retrieve data from MongoDB
        }),
      });

      const result = await response.json();

      if (response.ok) {
        setResults(result.result); // Set the results from the backend
        console.log('Query successful:', result.result);
      } else {
        setErrorMessage(result.error || 'Query processing failed.');
        console.error('Query failed:', result.error);
      }
    } catch (error) {
      setErrorMessage('Network error or server unreachable.');
      console.error('Query error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle form submission for NLP query
  const handleQuerySubmit = (e) => {
    e.preventDefault();
    if (query.trim() && currentDatasetId) {
      processNLPQuery(query);
    }
  };

  // Helper for generating UUIDs
  function uuidv4() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  // Render charts based on the results from the backend
  const renderChart = () => {
    if (!results || results.query_type === 'error' || results.chart_type === 'table') return null;

    const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1', '#a4de6c', '#d0ed57', '#83a6ed', '#8dd1e1', '#a4de6c'];

    switch (results.chart_type) {
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={results.data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={results.x_axis} />
              <YAxis />
              <Tooltip formatter={(value) => [`$${value.toLocaleString()}`, results.y_axis]} />
              <Bar dataKey={results.y_axis} fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        );
      case 'line':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={results.data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={results.x_axis} />
              <YAxis />
              <Tooltip formatter={(value) => [`$${value.toLocaleString()}`, results.y_axis]} />
              <Line type="monotone" dataKey={results.y_axis} stroke="#8884d8" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        );
      case 'pie':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={results.data}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value" // Pie chart typically uses 'value' for its slices
              >
                {results.data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => [`$${value.toLocaleString()}`, 'Value']} />
            </PieChart>
          </ResponsiveContainer>
        );
      default:
        return null;
    }
  };

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(to bottom right, #e0f2fe, #e8eaf6)', padding: '24px', fontFamily: 'sans-serif' }}>
      <div style={{ maxWidth: '960px', margin: '0 auto' }}>
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <h1 style={{ fontSize: '36px', fontWeight: 'bold', color: '#333', marginBottom: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
            <Sparkles style={{ color: '#4f46e5' }} size={24} />
            NLP Business Intelligence
          </h1>
          <p style={{ color: '#555', fontSize: '18px' }}>Upload your data and ask questions in natural language</p>
          {userId && (
            <p style={{ fontSize: '14px', color: '#666', marginTop: '8px' }}>
              Current User ID: <span style={{ fontFamily: 'monospace', color: '#4f46e5' }}>{userId}</span>
            </p>
          )}
        </div>

        {/* Error Message Display */}
        {errorMessage && (
          <div style={{ background: '#fee2e2', border: '1px solid #ef4444', color: '#b91c1c', padding: '16px', borderRadius: '8px', position: 'relative', marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <XCircle size={20} style={{ flexShrink: 0 }} />
            <span style={{ display: 'block' }}>{errorMessage}</span>
          </div>
        )}

        {/* File Upload Section */}
        <div style={{ background: '#fff', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)', padding: '32px', marginBottom: '32px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
            <Upload style={{ color: '#4f46e5' }} size={24} />
            <h2 style={{ fontSize: '24px', fontWeight: 'semibold', color: '#333' }}>Upload Data File</h2>
          </div>
          
          <div style={{ border: '2px dashed #ccc', borderRadius: '8px', padding: '32px', textAlign: 'center', transition: 'border-color 0.3s' }}>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept=".csv,.xlsx,.xls,.pdf"
              style={{ display: 'none' }}
            />
            
            {uploadStatus === 'processing' && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', color: '#2563eb' }}>
                <div style={{ animation: 'spin 1s linear infinite', borderRadius: '50%', height: '20px', width: '20px', borderBottom: '2px solid #2563eb' }}></div>
                <span>Processing file...</span>
              </div>
            )}
            
            {uploadStatus === 'success' && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', color: '#16a34a', marginBottom: '16px' }}>
                <CheckCircle size={20} />
                <span>File uploaded successfully! Dataset ID: {currentDatasetId}</span>
              </div>
            )}
            
            {!uploadedFile || uploadStatus === 'error' ? (
              <div>
                <FileText style={{ margin: '0 auto', color: '#999', marginBottom: '16px' }} size={48} />
                <p style={{ color: '#555', marginBottom: '16px' }}>Drop your CSV, Excel, or PDF file here</p>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  style={{ background: '#4f46e5', color: '#fff', padding: '10px 24px', borderRadius: '8px', border: 'none', cursor: 'pointer', transition: 'background-color 0.3s' }}
                  onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#4338ca'}
                  onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#4f46e5'}
                >
                  Choose File
                </button>
                <p style={{ fontSize: '14px', color: '#666', marginTop: '8px' }}>Supports CSV, Excel (.xlsx, .xls), and PDF tables</p>
              </div>
            ) : (
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', color: '#444', marginBottom: '16px' }}>
                  <FileText size={20} />
                  <span style={{ fontWeight: 'medium' }}>{uploadedFile.name}</span>
                </div>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  style={{ color: '#4f46e5', background: 'none', border: 'none', cursor: 'pointer', transition: 'color 0.3s' }}
                  onMouseOver={(e) => e.currentTarget.style.color = '#3730a3'}
                  onMouseOut={(e) => e.currentTarget.style.color = '#4f46e5'}
                >
                  Choose Different File
                </button>
              </div>
            )}
          </div>
          
          {data.length > 0 && (
            <div style={{ marginTop: '24px', padding: '16px', background: '#f8f8f8', borderRadius: '8px' }}>
              <h3 style={{ fontWeight: 'semibold', color: '#444', marginBottom: '8px' }}>Data Preview</h3>
              <p style={{ fontSize: '14px', color: '#666', marginBottom: '12px' }}>
                {data.length} records • Columns: {columns.join(', ')}
              </p>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ minWidth: '100%', fontSize: '14px', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ background: '#eee' }}>
                      {columns.map(col => (
                        <th key={col} style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 'medium', color: '#444', textTransform: 'capitalize' }}>
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.slice(0, 3).map((row, idx) => (
                      <tr key={idx} style={{ borderTop: '1px solid #eee' }}>
                        {columns.map(col => (
                          <td key={col} style={{ padding: '8px 12px', color: '#666' }}>
                            {typeof row[col] === 'number' && col.toLowerCase().includes('sales') 
                              ? `$${row[col].toLocaleString()}` 
                              : row[col]}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Query Section */}
        {currentDatasetId && (
          <div style={{ background: '#fff', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)', padding: '32px', marginBottom: '32px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
              <BarChart3 style={{ color: '#4f46e5' }} size={24} />
              <h2 style={{ fontSize: '24px', fontWeight: 'semibold', color: '#333' }}>Ask Questions About Your Data</h2>
            </div>
            
            <div style={{ marginBottom: '24px' }}>
              <div style={{ display: 'flex', gap: '12px' }}>
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="e.g., 'Show me the top products by sales' or 'What's the trend over time?'"
                  style={{ flex: '1', padding: '12px 16px', border: '1px solid #ccc', borderRadius: '8px', outline: 'none' }}
                  onFocus={(e) => e.target.style.borderColor = '#4f46e5'}
                  onBlur={(e) => e.target.style.borderColor = '#ccc'}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' && query.trim()) {
                      handleQuerySubmit(e);
                    }
                  }}
                />
                <button
                  onClick={handleQuerySubmit}
                  disabled={isLoading || !query.trim()}
                  style={{ background: '#4f46e5', color: '#fff', padding: '12px 24px', borderRadius: '8px', border: 'none', cursor: 'pointer', transition: 'background-color 0.3s', opacity: (isLoading || !query.trim()) ? 0.5 : 1 }}
                  onMouseOver={(e) => e.currentTarget.style.backgroundColor = (isLoading || !query.trim()) ? '' : '#4338ca'}
                  onMouseOut={(e) => e.currentTarget.style.backgroundColor = (isLoading || !query.trim()) ? '' : '#4f46e5'}
                >
                  {isLoading ? 'Analyzing...' : 'Ask'}
                </button>
              </div>
            </div>
            
            <div style={{ marginBottom: '24px' }}>
              <h3 style={{ fontSize: '14px', fontWeight: 'medium', color: '#444', marginBottom: '12px' }}>Try these example queries:</h3>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {suggestedQueries.map((suggested, idx) => (
                  <button
                    key={idx}
                    onClick={() => setQuery(suggested)}
                    style={{ padding: '6px 12px', background: '#eee', color: '#444', borderRadius: '9999px', fontSize: '14px', border: 'none', cursor: 'pointer', transition: 'background-color 0.3s' }}
                    onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#ddd'}
                    onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#eee'}
                  >
                    {suggested}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Results Section */}
        {results && results.query_type !== 'error' && (
          <div style={{ background: '#fff', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)', padding: '32px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px' }}>
              <h2 style={{ fontSize: '24px', fontWeight: 'semibold', color: '#333' }}>{results.title}</h2>
              {/* Export button - functionality to be added in Phase 2 */}
              <button style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 16px', background: '#eee', color: '#444', borderRadius: '8px', border: 'none', cursor: 'pointer', transition: 'background-color 0.3s' }}
                onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#ddd'}
                onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#eee'}
              >
                <Download size={16} />
                Export
              </button>
            </div>
            
            {results.insight && (
              <div style={{ marginBottom: '24px', padding: '16px', background: '#e0f2fe', borderLeft: '4px solid #60a5fa', borderRadius: '0 8px 8px 0' }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                  <AlertCircle style={{ color: '#2563eb', marginTop: '2px', flexShrink: 0 }} size={16} />
                  <p style={{ color: '#1d4ed8' }}>{results.insight}</p>
                </div>
              </div>
            )}
            
            {results.chart_type !== 'table' && renderChart()}
            
            {results.chart_type === 'table' && results.data && (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ minWidth: '100%', fontSize: '14px', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ background: '#eee' }}>
                      {results.data.length > 0 && Object.keys(results.data[0]).map(key => (
                        <th key={key} style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 'medium', color: '#444', textTransform: 'capitalize' }}>
                          {key.replace(/_/g, ' ')}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {results.data.map((row, idx) => (
                      <tr key={idx} style={{ borderTop: '1px solid #eee' }}>
                        {Object.entries(row).map(([key, value]) => (
                          <td key={key} style={{ padding: '8px 12px', color: '#666' }}>
                            {typeof value === 'number' ? value.toLocaleString() : value}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
      {/* Basic CSS for spin animation */}
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default NLPBusinessIntelligence;
