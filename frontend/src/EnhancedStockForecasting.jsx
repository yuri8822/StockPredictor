import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Button,
  Card,
  CardContent,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
  CircularProgress,
  Chip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  TextField,
} from '@mui/material';
import {
  TrendingUp,
  Assessment,
  AccountBalance,
  ShowChart,
  Refresh,
  PlayArrow,
} from '@mui/icons-material';
import axios from 'axios';
import Plot from 'react-plotly.js';

const EnhancedStockForecasting = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [ticker, setTicker] = useState('AAPL');
  const [forecastData, setForecastData] = useState(null);
  const [portfolioData, setPortfolioData] = useState(null);
  const [dashboardData, setDashboardData] = useState(null);
  const [candlestickData, setCandlestickData] = useState(null);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const API_BASE = 'http://localhost:5000/api';

  // Tab change handler
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
    setError(null);
  };

  // Fetch forecasting data
  const fetchForecast = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_BASE}/forecast`, {
        ticker: ticker.toUpperCase(),
        horizon: '24hrs',
        days: 90,
      });
      setForecastData(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to fetch forecast');
    } finally {
      setLoading(false);
    }
  };

  // Fetch portfolio data
  const fetchPortfolio = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_BASE}/portfolio/status`);
      setPortfolioData(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to fetch portfolio');
    } finally {
      setLoading(false);
    }
  };

  // Fetch evaluation dashboard
  const fetchDashboard = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_BASE}/evaluation/dashboard/${ticker}?days=30`);
      setDashboardData(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to fetch dashboard');
    } finally {
      setLoading(false);
    }
  };

  // Fetch candlestick chart with errors
  const fetchCandlestick = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_BASE}/candlestick/${ticker}?days=90&horizon=24hrs`);
      setCandlestickData(response.data.chart);
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to fetch candlestick');
    } finally {
      setLoading(false);
    }
  };

  // Trigger adaptive update
  const triggerAdaptiveUpdate = async (modelType) => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_BASE}/adaptive/trigger-update`, {
        ticker: ticker.toUpperCase(),
        model_type: modelType,
        days: 30,
      });
      alert(`Model updated successfully! MAE: ${response.data.metrics.mae.toFixed(6)}`);
      fetchDashboard(); // Refresh dashboard
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to update model');
    } finally {
      setLoading(false);
    }
  };

  // Execute trade
  const executeTrade = async (action, quantity) => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_BASE}/portfolio/trade`, {
        ticker: ticker.toUpperCase(),
        action,
        quantity,
      });
      alert(response.data.message);
      fetchPortfolio(); // Refresh portfolio
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to execute trade');
    } finally {
      setLoading(false);
    }
  };

  // Generate and execute trading signal
  const generateSignal = async (strategy) => {
    if (!forecastData || !forecastData.predictions || !forecastData.predictions.ensemble) {
      alert('Please generate a forecast first');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const prediction = forecastData.predictions.ensemble[0];
      const response = await axios.post(`${API_BASE}/portfolio/signal`, {
        ticker: ticker.toUpperCase(),
        prediction,
        strategy,
      });
      alert(`Action taken: ${response.data.action_taken.toUpperCase()}`);
      fetchPortfolio(); // Refresh portfolio
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to generate signal');
    } finally {
      setLoading(false);
    }
  };

  // Auto-load data when tab changes
  useEffect(() => {
    if (activeTab === 0 && ticker) {
      fetchCandlestick();
    } else if (activeTab === 1 && ticker) {
      fetchDashboard();
    } else if (activeTab === 2) {
      fetchPortfolio();
    }
  }, [activeTab, ticker]);

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Typography variant="h3" gutterBottom align="center" sx={{ mb: 4, fontWeight: 'bold' }}>
        🚀 Adaptive Stock Forecasting & Portfolio Management
      </Typography>

      {/* Ticker Selection */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Stock Ticker"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="AAPL"
              variant="outlined"
            />
          </Grid>
          <Grid item xs={12} md={8}>
            <Box sx={{ display: 'flex', gap: 2 }}>
              <Button
                variant="contained"
                startIcon={<PlayArrow />}
                onClick={fetchForecast}
                disabled={loading}
              >
                Generate Forecast
              </Button>
              <Button
                variant="outlined"
                startIcon={<Refresh />}
                onClick={() => {
                  if (activeTab === 0) fetchCandlestick();
                  else if (activeTab === 1) fetchDashboard();
                  else if (activeTab === 2) fetchPortfolio();
                }}
                disabled={loading}
              >
                Refresh
              </Button>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={activeTab} onChange={handleTabChange} variant="fullWidth">
          <Tab label="Candlestick & Predictions" icon={<ShowChart />} />
          <Tab label="Evaluation Dashboard" icon={<Assessment />} />
          <Tab label="Portfolio Management" icon={<AccountBalance />} />
        </Tabs>
      </Paper>

      {/* Tab Content */}
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
          <CircularProgress size={60} />
        </Box>
      )}

      {/* Tab 0: Candlestick Chart */}
      {activeTab === 0 && !loading && (
        <Paper sx={{ p: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h5">
              📊 Candlestick Chart with Prediction Errors
            </Typography>
            {lastUpdated && (
              <Chip 
                label={`Last updated: ${lastUpdated}`} 
                color="info" 
                size="small" 
                icon={<Refresh />}
              />
            )}
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            The chart shows historical price data with future predictions. Error bands indicate 
            prediction uncertainty (standard deviation across models). Yellow annotations show 
            prediction deviation from the last known price.
          </Typography>
          {candlestickData ? (
            <>
              <Plot
                data={candlestickData.data}
                layout={{
                  ...candlestickData.layout,
                  autosize: true,
                  width: undefined,
                }}
                useResizeHandler
                style={{ width: '100%', height: '600px' }}
              />
              <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap', justifyContent: 'center' }}>
                <Chip label="🟢 Actual Prices (Candlestick)" size="small" />
                <Chip label="🔴 Ensemble Forecast" color="error" size="small" />
                <Chip label="📊 Uncertainty Band" color="default" size="small" />
                <Chip label="🟡 Error Annotations" color="warning" size="small" />
                <Chip label="🟢 LSTM" color="success" size="small" />
                <Chip label="🔵 GRU" color="primary" size="small" />
                <Chip label="🟠 ARIMA" color="default" size="small" />
              </Box>
            </>
          ) : (
            <Typography color="text.secondary" align="center" sx={{ py: 4 }}>
              Click "Generate Forecast" or "Refresh" to view chart
            </Typography>
          )}
        </Paper>
      )}

      {/* Tab 1: Evaluation Dashboard */}
      {activeTab === 1 && !loading && (
        <Grid container spacing={3}>
          {/* Summary Cards */}
          {dashboardData && dashboardData.summary && (
            <>
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography color="text.secondary" gutterBottom>
                      Best Model
                    </Typography>
                    <Typography variant="h4">
                      {dashboardData.summary.best_model || 'N/A'}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography color="text.secondary" gutterBottom>
                      Overall MAE
                    </Typography>
                    <Typography variant="h4">
                      {dashboardData.summary.overall_mae?.toFixed(4) || 'N/A'}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography color="text.secondary" gutterBottom>
                      Overall MAPE
                    </Typography>
                    <Typography variant="h4">
                      {dashboardData.summary.overall_mape?.toFixed(2)}%
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </>
          )}

          {/* Model Performance */}
          {dashboardData && dashboardData.model_performance && (
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h5" gutterBottom>
                  Model Performance Comparison
                </Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Model</TableCell>
                        <TableCell align="right">Evaluations</TableCell>
                        <TableCell align="right">Avg MAE</TableCell>
                        <TableCell align="right">Avg RMSE</TableCell>
                        <TableCell align="right">Avg MAPE</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(dashboardData.model_performance).map(([model, stats]) => (
                        <TableRow key={model}>
                          <TableCell>
                            <Chip label={model} color="primary" />
                          </TableCell>
                          <TableCell align="right">{stats.num_evaluations}</TableCell>
                          <TableCell align="right">{stats.mae_mean.toFixed(6)}</TableCell>
                          <TableCell align="right">{stats.rmse_mean.toFixed(6)}</TableCell>
                          <TableCell align="right">{stats.mape_mean.toFixed(2)}%</TableCell>
                          <TableCell>
                            <Button
                              size="small"
                              variant="outlined"
                              onClick={() => triggerAdaptiveUpdate(model)}
                            >
                              Update Model
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            </Grid>
          )}

          {/* Alerts */}
          {dashboardData && dashboardData.alerts && dashboardData.alerts.length > 0 && (
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h5" gutterBottom>
                  ⚠️ Performance Alerts
                </Typography>
                {dashboardData.alerts.map((alert, idx) => (
                  <Alert key={idx} severity={alert.severity} sx={{ mb: 2 }}>
                    <strong>{alert.model_type}:</strong> {alert.message}
                  </Alert>
                ))}
              </Paper>
            </Grid>
          )}

          {/* Metrics Time Series */}
          {dashboardData && dashboardData.time_series && dashboardData.time_series.length > 0 && (
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h5" gutterBottom>
                  Metrics Over Time
                </Typography>
                <Plot
                  data={[
                    {
                      x: dashboardData.time_series.map((t) => t.timestamp),
                      y: dashboardData.time_series.map((t) => t.mae),
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: 'MAE',
                      line: { color: 'blue' },
                    },
                    {
                      x: dashboardData.time_series.map((t) => t.timestamp),
                      y: dashboardData.time_series.map((t) => t.rmse),
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: 'RMSE',
                      line: { color: 'red' },
                    },
                  ]}
                  layout={{
                    title: 'Model Performance Trends',
                    xaxis: { title: 'Time' },
                    yaxis: { title: 'Error Metric' },
                    height: 400,
                    autosize: true,
                  }}
                  useResizeHandler
                  style={{ width: '100%' }}
                />
              </Paper>
            </Grid>
          )}
        </Grid>
      )}

      {/* Tab 2: Portfolio Management */}
      {activeTab === 2 && !loading && (
        <Grid container spacing={3}>
          {/* Portfolio Metrics */}
          {portfolioData && portfolioData.metrics && (
            <>
              <Grid item xs={12} md={3}>
                <Card sx={{ bgcolor: 'primary.light', color: 'white' }}>
                  <CardContent>
                    <Typography color="inherit" gutterBottom>
                      Total Value
                    </Typography>
                    <Typography variant="h4" color="inherit">
                      ${portfolioData.metrics.total_value.toFixed(2)}
                    </Typography>
                    <Typography variant="body2" color="inherit">
                      Return: {portfolioData.metrics.total_return_pct.toFixed(2)}%
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={3}>
                <Card>
                  <CardContent>
                    <Typography color="text.secondary" gutterBottom>
                      Cash
                    </Typography>
                    <Typography variant="h4">
                      ${portfolioData.metrics.cash.toFixed(2)}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={3}>
                <Card>
                  <CardContent>
                    <Typography color="text.secondary" gutterBottom>
                      Sharpe Ratio
                    </Typography>
                    <Typography variant="h4">
                      {portfolioData.metrics.sharpe_ratio.toFixed(3)}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={3}>
                <Card>
                  <CardContent>
                    <Typography color="text.secondary" gutterBottom>
                      Win Rate
                    </Typography>
                    <Typography variant="h4">
                      {portfolioData.metrics.win_rate_pct.toFixed(1)}%
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </>
          )}

          {/* Trading Actions */}
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h5" gutterBottom>
                Trading Actions
              </Typography>
              <Grid container spacing={2}>
                <Grid item>
                  <Button
                    variant="contained"
                    color="success"
                    onClick={() => executeTrade('buy', 10)}
                    disabled={loading}
                  >
                    Buy 10 Shares
                  </Button>
                </Grid>
                <Grid item>
                  <Button
                    variant="contained"
                    color="error"
                    onClick={() => executeTrade('sell', 10)}
                    disabled={loading}
                  >
                    Sell 10 Shares
                  </Button>
                </Grid>
                <Grid item>
                  <Button
                    variant="outlined"
                    onClick={() => generateSignal('simple')}
                    disabled={loading}
                  >
                    Auto-Trade (Simple Strategy)
                  </Button>
                </Grid>
                <Grid item>
                  <Button
                    variant="outlined"
                    onClick={() => generateSignal('momentum')}
                    disabled={loading}
                  >
                    Auto-Trade (Momentum Strategy)
                  </Button>
                </Grid>
              </Grid>
            </Paper>
          </Grid>

          {/* Current Positions */}
          {portfolioData && portfolioData.positions && portfolioData.positions.length > 0 && (
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h5" gutterBottom>
                  Current Positions
                </Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Ticker</TableCell>
                        <TableCell align="right">Quantity</TableCell>
                        <TableCell align="right">Entry Price</TableCell>
                        <TableCell align="right">Current Price</TableCell>
                        <TableCell align="right">P&L</TableCell>
                        <TableCell align="right">Return</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {portfolioData.positions.map((pos) => (
                        <TableRow key={pos.ticker}>
                          <TableCell>
                            <Chip label={pos.ticker} color="primary" />
                          </TableCell>
                          <TableCell align="right">{pos.quantity.toFixed(2)}</TableCell>
                          <TableCell align="right">${pos.entry_price.toFixed(2)}</TableCell>
                          <TableCell align="right">${pos.current_price.toFixed(2)}</TableCell>
                          <TableCell
                            align="right"
                            sx={{ color: pos.unrealized_pnl >= 0 ? 'success.main' : 'error.main' }}
                          >
                            ${pos.unrealized_pnl.toFixed(2)}
                          </TableCell>
                          <TableCell
                            align="right"
                            sx={{ color: pos.return_pct >= 0 ? 'success.main' : 'error.main' }}
                          >
                            {pos.return_pct.toFixed(2)}%
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            </Grid>
          )}
        </Grid>
      )}
    </Container>
  );
};

export default EnhancedStockForecasting;
