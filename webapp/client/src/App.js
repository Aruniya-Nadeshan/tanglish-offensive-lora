import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const LABEL_COLORS = {
  'Not_offensive': '#22c55e',
  'Offensive_Untargetede': '#f97316',
  'Offensive_Targeted_Insult_Individual': '#ef4444',
  'Offensive_Targeted_Insult_Group': '#dc2626',
  'Offensive_Targeted_Insult_Other': '#b91c1c',
  'not-Tamil': '#6b7280'
};

function App() {
  const [comment, setComment] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const classify = async () => {
    if (!comment.trim()) return;
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await axios.post('http://localhost:5000/classify', { comment });
      setResult(response.data);
    } catch (err) {
      setError('Classification failed. Make sure the server is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Tanglish Offensive Language Classifier</h1>
        <p>Classifies Tamil-English code-mixed comments into 6 categories</p>
      </header>

      <main className="main">
        <div className="input-section">
          <label>Enter a Tanglish comment</label>
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="e.g. Vera level BGM .. semma trailer"
            rows={4}
          />
          <button onClick={classify} disabled={loading || !comment.trim()}>
            {loading ? 'Classifying...' : 'Classify'}
          </button>
        </div>

        {error && <div className="error">{error}</div>}

        {result && (
          <div className="result-section">
            <div className="prediction">
              <span className="prediction-label">Prediction</span>
              <span
                className="prediction-value"
                style={{ color: LABEL_COLORS[result.prediction] }}
              >
                {result.prediction}
              </span>
            </div>

            <div className="scores">
              <h3>Confidence scores</h3>
              {result.scores.map(({ label, score }) => (
                <div className="score-row" key={label}>
                  <span className="score-label">{label}</span>
                  <div className="bar-wrap">
                    <div
                      className="bar"
                      style={{
                        width: `${(score * 100).toFixed(1)}%`,
                        background: LABEL_COLORS[label]
                      }}
                    />
                  </div>
                  <span className="score-value">{(score * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
