import React, { useState } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  TextField,
  MenuItem,
  Button,
  Alert,
  CircularProgress,
  Card,
  CardContent,
} from '@mui/material';
import { TrendingUp, Analytics, Timeline } from '@mui/icons-material';
import axios from 'axios';
import Plot from 'react-plotly.js';

const StockForecasting = () => {
  const [formData, setFormData] = useState({
    ticker: 'AAPL',
    horizon: '24hrs',
    days: 90
  });
  
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const horizonOptions = [
    { value: '1hr', label: '1 Day Ahead' },
    { value: '3hrs', label: '3 Days Ahead' },
    { value: '24hrs', label: '24 Days Ahead' },
    { value: '72hrs', label: '72 Days Ahead' }
  ];

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await axios.post('http://localhost:5000/api/forecast', {
        ticker: formData.ticker.toUpperCase(),
        horizon: formData.horizon,
        days: parseInt(formData.days)
      });

      console.log('Received response:', response.data);
      console.log('Metrics available:', response.data?.metrics ? Object.keys(response.data.metrics) : 'None');
      console.log('Chart available:', response.data?.chart ? 'Yes' : 'No');
      console.log('Full response structure:', {
        hasChart: !!response.data?.chart,
        hasMetrics: !!response.data?.metrics,
        hasDatasetInfo: !!response.data?.dataset_info,
        hasPredictions: !!response.data?.predictions,
        chartDataLength: response.data?.chart?.data?.length,
        metricsKeys: response.data?.metrics ? Object.keys(response.data.metrics) : []
      });
      
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred while generating the forecast');
    } finally {
      setLoading(false);
    }
  };

  const renderMetrics = () => {
    if (!results?.metrics) {
      console.log('No metrics found in results:', results);
      return null;
    }

    console.log('Rendering metrics for models:', Object.keys(results.metrics));
    const models = ['arima', 'lstm', 'ensemble'];
    
    return (
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {models.map(model => (
          <Grid item xs={12} md={4} key={model}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ 
                  textTransform: 'uppercase',
                  color: 'primary.main',
                  fontWeight: 'bold'
                }}>
                  {model} Model
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">RMSE:</Typography>
                    <Typography variant="body2" fontWeight="bold">
                      ${results.metrics[model]?.rmse?.toFixed(4) || 'N/A'}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">MAE:</Typography>
                    <Typography variant="body2" fontWeight="bold">
                      ${results.metrics[model]?.mae?.toFixed(4) || 'N/A'}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">MAPE:</Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {results.metrics[model]?.mape?.toFixed(2) || 'N/A'}%
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    );
  };



  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        <Typography variant="h2" component="h1" gutterBottom sx={{ 
          fontWeight: 'bold',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          mb: 2
        }}>
          📈 Stock Forecasting Application
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 4 }}>
          ML-powered forecasting using curated datasets (ARIMA + LSTM + Ensemble)
        </Typography>
        
        <Alert severity="info" sx={{ maxWidth: 800, mx: 'auto' }}>
          <Typography variant="body2">
            <strong>ℹ️ Data Integration:</strong> This application uses curated datasets generated by{' '}
            <code>StockDataCollector.py</code>. When you submit a forecast request, it will automatically 
            run the data collector to fetch the latest data with technical indicators and sentiment analysis.
          </Typography>
        </Alert>
      </Box>

      {/* Form */}
      <Paper sx={{ p: 4, mb: 4 }}>
        <Box component="form" onSubmit={handleSubmit}>
          <Grid container spacing={3} alignItems="end">
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                fullWidth
                label="Ticker Symbol"
                name="ticker"
                value={formData.ticker}
                onChange={handleInputChange}
                required
                variant="outlined"
              />
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                fullWidth
                select
                label="Forecast Horizon"
                name="horizon"
                value={formData.horizon}
                onChange={handleInputChange}
                variant="outlined"
              >
                {horizonOptions.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                fullWidth
                type="number"
                label="Historical Days"
                name="days"
                value={formData.days}
                onChange={handleInputChange}
                inputProps={{ min: 30, max: 365 }}
                variant="outlined"
              />
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Button
                type="submit"
                variant="contained"
                size="large"
                fullWidth
                disabled={loading}
                startIcon={loading ? <CircularProgress size={20} /> : <TrendingUp />}
                sx={{
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  '&:hover': {
                    background: 'linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%)',
                  }
                }}
              >
                {loading ? 'Generating...' : 'Generate Forecast'}
              </Button>
            </Grid>
          </Grid>
        </Box>
      </Paper>

      {/* Loading */}
      {loading && (
        <Paper sx={{ p: 4, textAlign: 'center', mb: 4 }}>
          <CircularProgress size={60} sx={{ mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            Running StockDataCollector.py and training models...
          </Typography>
          <Typography color="text.secondary">
            This may take 1-2 minutes.
          </Typography>
        </Paper>
      )}

      {/* Error */}
      {error && (
        <Alert severity="error" sx={{ mb: 4 }}>
          {error}
        </Alert>
      )}

      {/* Debug Info */}
      {results && (
        <Alert severity="info" sx={{ mb: 4 }}>
          <Typography variant="body2">
            <strong>Debug Info:</strong> Chart: {results.chart ? '✅' : '❌'} | 
            Metrics: {results.metrics ? '✅' : '❌'} | 
            Dataset: {results.dataset_info ? '✅' : '❌'} | 
            Chart Traces: {results.chart?.data?.length || 0}
          </Typography>
        </Alert>
      )}

      {/* Results */}
      {results && (
        <>
          {/* Dataset Info */}
          {results.dataset_info && (
            <Paper sx={{ p: 3, mb: 4, backgroundColor: '#fff3cd', borderLeft: '4px solid #ffc107' }}>
              <Typography variant="h6" sx={{ color: '#856404', mb: 1 }}>
                📊 Curated Dataset Information
              </Typography>
              <Typography sx={{ color: '#856404' }}>
                Loaded <strong>{results.dataset_info.num_rows}</strong> rows of data.{' '}
                Date range: <strong>{results.dataset_info.date_range}</strong>.{' '}
                Features included: {results.dataset_info.features.join(', ')}
              </Typography>
            </Paper>
          )}

          {/* Metrics */}
          <Paper sx={{ p: 3, mb: 4 }}>
            <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Analytics sx={{ mr: 1 }} />
              Model Performance Metrics
            </Typography>
            {results.metrics ? renderMetrics() : (
              <Alert severity="warning">No metrics available in the response</Alert>
            )}
          </Paper>

          {/* Chart */}
          <Paper sx={{ p: 3, mb: 4 }}>
            <Typography variant="h6" gutterBottom>
              <Timeline sx={{ mr: 1 }} />
              Stock Price Forecast Chart
            </Typography>
            {results.chart ? (
              <Box sx={{ height: 800, width: '100%' }}>
                <Plot
                  data={results.chart.data}
                  layout={results.chart.layout}
                  style={{ width: '100%', height: '100%' }}
                  config={{ 
                    displayModeBar: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['pan2d', 'lasso2d']
                  }}
                />
              </Box>
            ) : (
              <Alert severity="warning">No chart data available in the response</Alert>
            )}
          </Paper>
        </>
      )}
    </Container>
  );
};

export default StockForecasting;