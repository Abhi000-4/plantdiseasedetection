import React, { useState, useEffect } from 'react';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('predict');
  const [blockchainHistory, setBlockchainHistory] = useState([]);
  const [blockchainStats, setBlockchainStats] = useState(null);
  const [verificationStatus, setVerificationStatus] = useState(null);

  useEffect(() => {
    if (activeTab === 'blockchain') {
      fetchBlockchainData();
    }
  }, [activeTab]);

  const fetchBlockchainData = async () => {
    try {
      const [historyRes, statsRes, verifyRes] = await Promise.all([
        fetch('http://127.0.0.1:5000/blockchain/history'),
        fetch('http://127.0.0.1:5000/blockchain/stats'),
        fetch('http://127.0.0.1:5000/blockchain/verify')
      ]);

      const history = await historyRes.json();
      const stats = await statsRes.json();
      const verify = await verifyRes.json();

      setBlockchainHistory(history.predictions || []);
      setBlockchainStats(stats);
      setVerificationStatus(verify);
    } catch (err) {
      console.error('Error fetching blockchain data:', err);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Server returned ${response.status}`);
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message || 'Failed to get prediction. Please check if the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setPrediction(null);
    setError(null);
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const truncateHash = (hash) => {
    if (!hash) return 'N/A';
    return `${hash.substring(0, 8)}...${hash.substring(hash.length - 8)}`;
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '40px 20px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        background: 'white',
        borderRadius: '20px',
        padding: '40px',
        boxShadow: '0 20px 60px rgba(0,0,0,0.3)'
      }}>
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '30px' }}>
          <h1 style={{
            fontSize: '32px',
            fontWeight: '700',
            color: '#1a202c',
            marginBottom: '10px'
          }}>
            🌱 AI Crop Disease Detection
          </h1>
          <p style={{ color: '#718096', fontSize: '16px' }}>
            Upload soybean leaf images with blockchain traceability
          </p>
        </div>

        {/* Tab Navigation */}
        <div style={{
          display: 'flex',
          gap: '10px',
          marginBottom: '30px',
          borderBottom: '2px solid #e2e8f0'
        }}>
          <button
            onClick={() => setActiveTab('predict')}
            style={{
              padding: '12px 24px',
              fontSize: '16px',
              fontWeight: '600',
              color: activeTab === 'predict' ? '#667eea' : '#718096',
              background: 'transparent',
              border: 'none',
              borderBottom: activeTab === 'predict' ? '3px solid #667eea' : '3px solid transparent',
              cursor: 'pointer',
              transition: 'all 0.3s'
            }}
          >
            🔬 Predict
          </button>
          <button
            onClick={() => setActiveTab('blockchain')}
            style={{
              padding: '12px 24px',
              fontSize: '16px',
              fontWeight: '600',
              color: activeTab === 'blockchain' ? '#667eea' : '#718096',
              background: 'transparent',
              border: 'none',
              borderBottom: activeTab === 'blockchain' ? '3px solid #667eea' : '3px solid transparent',
              cursor: 'pointer',
              transition: 'all 0.3s'
            }}
          >
            🔗 Blockchain
          </button>
        </div>

        {/* Predict Tab */}
        {activeTab === 'predict' && (
          <div>
            {/* Upload Area */}
            <div style={{
              border: '3px dashed #cbd5e0',
              borderRadius: '12px',
              padding: '40px',
              textAlign: 'center',
              marginBottom: '30px',
              background: '#f7fafc',
              transition: 'all 0.3s',
              cursor: 'pointer'
            }}>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
                id="file-input"
              />
              <label htmlFor="file-input" style={{ cursor: 'pointer', display: 'block' }}>
                {preview ? (
                  <img
                    src={preview}
                    alt="Preview"
                    style={{
                      maxWidth: '100%',
                      maxHeight: '400px',
                      borderRadius: '8px',
                      marginBottom: '20px'
                    }}
                  />
                ) : (
                  <div>
                    <div style={{ fontSize: '48px', marginBottom: '15px' }}>📸</div>
                    <p style={{ fontSize: '18px', color: '#4a5568', marginBottom: '8px' }}>
                      Click to upload or drag and drop
                    </p>
                    <p style={{ fontSize: '14px', color: '#a0aec0' }}>
                      PNG, JPG or JPEG (MAX. 10MB)
                    </p>
                  </div>
                )}
              </label>
            </div>

            {/* Action Buttons */}
            <div style={{
              display: 'flex',
              gap: '15px',
              marginBottom: '30px',
              justifyContent: 'center'
            }}>
              <button
                onClick={handleUpload}
                disabled={!selectedFile || loading}
                style={{
                  padding: '14px 32px',
                  fontSize: '16px',
                  fontWeight: '600',
                  color: 'white',
                  background: selectedFile && !loading 
                    ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' 
                    : '#cbd5e0',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: selectedFile && !loading ? 'pointer' : 'not-allowed',
                  transition: 'all 0.3s',
                  boxShadow: selectedFile && !loading ? '0 4px 15px rgba(102, 126, 234, 0.4)' : 'none'
                }}
              >
                {loading ? '🔄 Analyzing...' : '🔬 Analyze Image'}
              </button>
              
              {(preview || prediction) && (
                <button
                  onClick={handleReset}
                  style={{
                    padding: '14px 32px',
                    fontSize: '16px',
                    fontWeight: '600',
                    color: '#4a5568',
                    background: 'white',
                    border: '2px solid #e2e8f0',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    transition: 'all 0.3s'
                  }}
                >
                  🔄 Reset
                </button>
              )}
            </div>

            {/* Error Message */}
            {error && (
              <div style={{
                padding: '16px',
                background: '#fed7d7',
                border: '1px solid #fc8181',
                borderRadius: '8px',
                color: '#c53030',
                marginBottom: '20px'
              }}>
                <strong>Error:</strong> {error}
              </div>
            )}

            {/* Prediction Results */}
            {prediction && (
              <div>
                <div style={{
                  background: 'linear-gradient(135deg, #e0f2fe 0%, #dbeafe 100%)',
                  borderRadius: '12px',
                  padding: '30px',
                  border: '2px solid #93c5fd',
                  marginBottom: '20px'
                }}>
                  <h2 style={{
                    fontSize: '24px',
                    fontWeight: '700',
                    color: '#1e40af',
                    marginBottom: '20px',
                    textAlign: 'center'
                  }}>
                    📊 Analysis Results
                  </h2>
                  
                  <div style={{ display: 'grid', gap: '15px' }}>
                    <div style={{
                      background: 'white',
                      padding: '20px',
                      borderRadius: '8px',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                    }}>
                      <div style={{ fontSize: '14px', color: '#6b7280', marginBottom: '5px' }}>
                        Disease Detected
                      </div>
                      <div style={{
                        fontSize: '24px',
                        fontWeight: '700',
                        color: prediction.disease === 'healthy' ? '#059669' : '#dc2626'
                      }}>
                        {prediction.disease.replace(/_/g, ' ').toUpperCase()}
                      </div>
                    </div>

                    {prediction.moisture !== undefined && (
                      <div style={{
                        background: 'white',
                        padding: '20px',
                        borderRadius: '8px',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                      }}>
                        <div style={{ fontSize: '14px', color: '#6b7280', marginBottom: '5px' }}>
                          Moisture Level
                        </div>
                        <div style={{
                          fontSize: '24px',
                          fontWeight: '700',
                          color: '#2563eb'
                        }}>
                          {prediction.moisture}%
                        </div>
                        <div style={{
                          width: '100%',
                          height: '8px',
                          background: '#e5e7eb',
                          borderRadius: '4px',
                          marginTop: '10px',
                          overflow: 'hidden'
                        }}>
                          <div style={{
                            width: `${prediction.moisture}%`,
                            height: '100%',
                            background: 'linear-gradient(90deg, #3b82f6 0%, #2563eb 100%)',
                            transition: 'width 0.5s'
                          }} />
                        </div>
                      </div>
                    )}

                    {prediction.confidence !== undefined && (
                      <div style={{
                        background: 'white',
                        padding: '20px',
                        borderRadius: '8px',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                      }}>
                        <div style={{ fontSize: '14px', color: '#6b7280', marginBottom: '5px' }}>
                          Confidence Score
                        </div>
                        <div style={{
                          fontSize: '24px',
                          fontWeight: '700',
                          color: '#059669'
                        }}>
                          {(prediction.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Blockchain Info */}
                {prediction.blockchain && (
                  <div style={{
                    background: 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)',
                    borderRadius: '12px',
                    padding: '25px',
                    border: '2px solid #fbbf24'
                  }}>
                    <h3 style={{
                      fontSize: '20px',
                      fontWeight: '700',
                      color: '#92400e',
                      marginBottom: '15px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '10px'
                    }}>
                      🔗 Blockchain Verification
                      {prediction.blockchain.recorded && (
                        <span style={{
                          fontSize: '12px',
                          fontWeight: '600',
                          color: '#059669',
                          background: '#d1fae5',
                          padding: '4px 12px',
                          borderRadius: '12px'
                        }}>
                          ✅ RECORDED
                        </span>
                      )}
                    </h3>
                    
                    {prediction.blockchain.recorded ? (
                      <div style={{ display: 'grid', gap: '10px' }}>
                        <div style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          padding: '10px',
                          background: 'rgba(255,255,255,0.5)',
                          borderRadius: '6px'
                        }}>
                          <span style={{ fontWeight: '600', color: '#92400e' }}>Block Index:</span>
                          <span style={{ color: '#78350f' }}>#{prediction.blockchain.block_index}</span>
                        </div>
                        <div style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          padding: '10px',
                          background: 'rgba(255,255,255,0.5)',
                          borderRadius: '6px'
                        }}>
                          <span style={{ fontWeight: '600', color: '#92400e' }}>Block Hash:</span>
                          <span style={{ 
                            color: '#78350f',
                            fontSize: '12px',
                            fontFamily: 'monospace'
                          }}>
                            {truncateHash(prediction.blockchain.block_hash)}
                          </span>
                        </div>
                        <div style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          padding: '10px',
                          background: 'rgba(255,255,255,0.5)',
                          borderRadius: '6px'
                        }}>
                          <span style={{ fontWeight: '600', color: '#92400e' }}>Timestamp:</span>
                          <span style={{ color: '#78350f', fontSize: '14px' }}>
                            {formatTimestamp(prediction.blockchain.timestamp)}
                          </span>
                        </div>
                        <div style={{
                          marginTop: '10px',
                          padding: '12px',
                          background: 'rgba(255,255,255,0.7)',
                          borderRadius: '6px',
                          fontSize: '13px',
                          color: '#92400e',
                          textAlign: 'center'
                        }}>
                          🔒 This prediction is permanently recorded on the blockchain and cannot be altered
                        </div>
                      </div>
                    ) : (
                      <div style={{
                        padding: '15px',
                        background: 'rgba(239, 68, 68, 0.1)',
                        borderRadius: '6px',
                        color: '#dc2626',
                        textAlign: 'center'
                      }}>
                        ⚠️ Failed to record on blockchain: {prediction.blockchain.error}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Blockchain Tab */}
        {activeTab === 'blockchain' && (
          <div>
            {/* Stats Cards */}
            {blockchainStats && (
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '20px',
                marginBottom: '30px'
              }}>
                <div style={{
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  padding: '20px',
                  borderRadius: '12px',
                  color: 'white',
                  boxShadow: '0 4px 15px rgba(102, 126, 234, 0.3)'
                }}>
                  <div style={{ fontSize: '14px', opacity: 0.9, marginBottom: '5px' }}>Total Blocks</div>
                  <div style={{ fontSize: '32px', fontWeight: '700' }}>{blockchainStats.total_blocks}</div>
                </div>
                
                <div style={{
                  background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                  padding: '20px',
                  borderRadius: '12px',
                  color: 'white',
                  boxShadow: '0 4px 15px rgba(240, 147, 251, 0.3)'
                }}>
                  <div style={{ fontSize: '14px', opacity: 0.9, marginBottom: '5px' }}>Predictions</div>
                  <div style={{ fontSize: '32px', fontWeight: '700' }}>{blockchainStats.total_predictions}</div>
                </div>
                
                <div style={{
                  background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                  padding: '20px',
                  borderRadius: '12px',
                  color: 'white',
                  boxShadow: '0 4px 15px rgba(79, 172, 254, 0.3)'
                }}>
                  <div style={{ fontSize: '14px', opacity: 0.9, marginBottom: '5px' }}>Avg Moisture</div>
                  <div style={{ fontSize: '32px', fontWeight: '700' }}>{blockchainStats.average_moisture}%</div>
                </div>
                
                <div style={{
                  background: blockchainStats.blockchain_valid 
                    ? 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)' 
                    : 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
                  padding: '20px',
                  borderRadius: '12px',
                  color: 'white',
                  boxShadow: '0 4px 15px rgba(67, 233, 123, 0.3)'
                }}>
                  <div style={{ fontSize: '14px', opacity: 0.9, marginBottom: '5px' }}>Chain Status</div>
                  <div style={{ fontSize: '24px', fontWeight: '700' }}>
                    {blockchainStats.blockchain_valid ? '✅ VALID' : '❌ INVALID'}
                  </div>
                </div>
              </div>
            )}

            {/* Verification Status */}
            {verificationStatus && (
              <div style={{
                padding: '20px',
                background: verificationStatus.valid 
                  ? 'linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)' 
                  : 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)',
                borderRadius: '12px',
                marginBottom: '30px',
                border: `2px solid ${verificationStatus.valid ? '#10b981' : '#ef4444'}`
              }}>
                <div style={{
                  fontSize: '18px',
                  fontWeight: '700',
                  color: verificationStatus.valid ? '#065f46' : '#991b1b',
                  marginBottom: '10px'
                }}>
                  {verificationStatus.message}
                </div>
                <div style={{
                  fontSize: '14px',
                  color: verificationStatus.valid ? '#047857' : '#b91c1c'
                }}>
                  Total Blocks: {verificationStatus.blocks} | Predictions: {verificationStatus.predictions}
                </div>
              </div>
            )}

            {/* Predictions History */}
            <div>
              <h2 style={{
                fontSize: '24px',
                fontWeight: '700',
                color: '#1a202c',
                marginBottom: '20px'
              }}>
                📋 Prediction History
              </h2>
              
              {blockchainHistory.length === 0 ? (
                <div style={{
                  padding: '60px',
                  textAlign: 'center',
                  background: '#f7fafc',
                  borderRadius: '12px',
                  color: '#718096'
                }}>
                  <div style={{ fontSize: '48px', marginBottom: '15px' }}>📭</div>
                  <div style={{ fontSize: '18px' }}>No predictions recorded yet</div>
                  <div style={{ fontSize: '14px', marginTop: '8px' }}>Upload an image to get started!</div>
                </div>
              ) : (
                <div style={{ display: 'grid', gap: '15px' }}>
                  {blockchainHistory.map((pred, index) => (
                    <div key={index} style={{
                      background: 'white',
                      padding: '20px',
                      borderRadius: '12px',
                      border: '2px solid #e2e8f0',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
                      transition: 'all 0.3s'
                    }}>
                      <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'start',
                        marginBottom: '15px'
                      }}>
                        <div>
                          <div style={{
                            fontSize: '18px',
                            fontWeight: '700',
                            color: '#1a202c',
                            marginBottom: '5px'
                          }}>
                            {pred.disease.replace(/_/g, ' ').toUpperCase()}
                          </div>
                          <div style={{
                            fontSize: '12px',
                            color: '#718096'
                          }}>
                            {formatTimestamp(pred.timestamp)}
                          </div>
                        </div>
                        <div style={{
                          background: '#e0f2fe',
                          color: '#0369a1',
                          padding: '6px 12px',
                          borderRadius: '6px',
                          fontSize: '12px',
                          fontWeight: '600'
                        }}>
                          Block #{pred.block_index}
                        </div>
                      </div>
                      
                      <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                        gap: '15px',
                        marginBottom: '15px'
                      }}>
                        <div>
                          <div style={{ fontSize: '12px', color: '#718096', marginBottom: '3px' }}>
                            Moisture Level
                          </div>
                          <div style={{ fontSize: '18px', fontWeight: '600', color: '#2563eb' }}>
                            {pred.moisture}%
                          </div>
                        </div>
                        <div>
                          <div style={{ fontSize: '12px', color: '#718096', marginBottom: '3px' }}>
                            Confidence
                          </div>
                          <div style={{ fontSize: '18px', fontWeight: '600', color: '#059669' }}>
                            {(pred.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>
                      
                      <div style={{
                        padding: '10px',
                        background: '#f7fafc',
                        borderRadius: '6px',
                        fontSize: '12px',
                        fontFamily: 'monospace',
                        color: '#4a5568',
                        wordBreak: 'break-all'
                      }}>
                        <strong>Hash:</strong> {pred.block_hash}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Footer */}
        <div style={{
          marginTop: '40px',
          paddingTop: '20px',
          borderTop: '1px solid #e2e8f0',
          textAlign: 'center',
          color: '#9ca3af',
          fontSize: '14px'
        }}>
          <p>🔒 Powered by AI & Blockchain Technology</p>
          <p style={{ marginTop: '5px', fontSize: '12px' }}>
            All predictions are immutably recorded on the blockchain for transparency and traceability
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;