import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Paper,
  Chip,
  LinearProgress,
  Grid,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  CheckCircle,
  RadioButtonUnchecked,
  Loop,
  ExpandMore,
  TrendingUp,
  Psychology,
  DataObject,
  Speed,
  AutoFixHigh,
} from '@mui/icons-material';

const AdaptiveLearningExplainer = ({ learningData }) => {
  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'active':
        return 'primary';
      case 'pending':
        return 'default';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle color="success" />;
      case 'active':
        return <Loop color="primary" />;
      case 'pending':
        return <RadioButtonUnchecked color="disabled" />;
      default:
        return <RadioButtonUnchecked />;
    }
  };

  if (!learningData || !learningData.learning_process) {
    return (
      <Alert severity="info">
        No adaptive learning data available. Generate a forecast to see adaptive learning in action!
      </Alert>
    );
  }

  const { learning_process, explanation, learning_status } = learningData;

  return (
    <Box sx={{ width: '100%' }}>
      {/* Overview Card */}
      <Card sx={{ mb: 3, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
        <CardContent>
          <Typography variant="h5" gutterBottom sx={{ color: 'white', display: 'flex', alignItems: 'center', gap: 1 }}>
            <Psychology /> Adaptive Learning System
          </Typography>
          <Typography variant="body1" sx={{ color: 'rgba(255,255,255,0.9)', mb: 2 }}>
            Our system continuously learns from new data to improve prediction accuracy over time.
          </Typography>
          <Grid container spacing={2}>
            <Grid size={{ xs: 12, md: 3 }}>
              <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'rgba(255,255,255,0.1)' }}>
                <Typography variant="h4" sx={{ color: 'white', fontWeight: 'bold' }}>
                  {learning_status?.total_model_versions || 0}
                </Typography>
                <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.8)' }}>
                  Model Updates
                </Typography>
              </Paper>
            </Grid>
            <Grid size={{ xs: 12, md: 3 }}>
              <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'rgba(255,255,255,0.1)' }}>
                <Typography variant="h4" sx={{ color: 'white', fontWeight: 'bold' }}>
                  {learning_status?.average_improvement?.toFixed(1) || 0}%
                </Typography>
                <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.8)' }}>
                  Avg Improvement
                </Typography>
              </Paper>
            </Grid>
            <Grid size={{ xs: 12, md: 3 }}>
              <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'rgba(255,255,255,0.1)' }}>
                <Typography variant="h4" sx={{ color: 'white', fontWeight: 'bold' }}>
                  {learning_status?.active_model_types || 0}
                </Typography>
                <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.8)' }}>
                  Active Models
                </Typography>
              </Paper>
            </Grid>
            <Grid size={{ xs: 12, md: 3 }}>
              <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'rgba(255,255,255,0.1)' }}>
                <Typography variant="h4" sx={{ color: 'white', fontWeight: 'bold' }}>
                  <CheckCircle />
                </Typography>
                <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.8)' }}>
                  Real-time
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Learning Process Stepper */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <DataObject color="primary" /> Step-by-Step Learning Process
          </Typography>
          <Stepper orientation="vertical" sx={{ mt: 2 }}>
            {learning_process.map((step, index) => (
              <Step key={index} active={step.status === 'active'} completed={step.status === 'completed'}>
                <StepLabel
                  StepIconComponent={() => getStatusIcon(step.status)}
                  optional={
                    <Chip
                      label={step.status}
                      size="small"
                      color={getStatusColor(step.status)}
                      sx={{ mt: 0.5 }}
                    />
                  }
                >
                  <Typography variant="subtitle1" fontWeight="bold">
                    Step {step.step}: {step.title}
                  </Typography>
                </StepLabel>
                <StepContent>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    {step.description}
                  </Typography>
                  <Paper sx={{ p: 2, bgcolor: 'background.default', border: '1px solid', borderColor: 'divider' }}>
                    <Typography variant="caption" fontWeight="medium">
                      📊 {step.details}
                    </Typography>
                  </Paper>
                  {step.status === 'active' && (
                    <Box sx={{ mt: 2 }}>
                      <LinearProgress color="primary" />
                    </Box>
                  )}
                </StepContent>
              </Step>
            ))}
          </Stepper>
        </CardContent>
      </Card>

      {/* How It Works Section */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Speed color="primary" /> How Adaptive Learning Works
          </Typography>
          <List>
            {explanation?.how_it_works?.map((item, index) => (
              <ListItem key={index}>
                <ListItemIcon>
                  <TrendingUp color="success" />
                </ListItemIcon>
                <ListItemText
                  primary={item}
                  primaryTypographyProps={{
                    variant: 'body2',
                    color: 'text.primary',
                  }}
                />
              </ListItem>
            ))}
          </List>
        </CardContent>
      </Card>

      {/* Adaptive Features Section */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AutoFixHigh color="primary" /> Key Adaptive Features
          </Typography>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            {explanation?.adaptive_features?.map((feature, index) => (
              <Grid size={{ xs: 12, md: 6 }} key={index}>
                <Paper
                  sx={{
                    p: 2,
                    height: '100%',
                    border: '1px solid',
                    borderColor: 'primary.light',
                    '&:hover': {
                      boxShadow: 3,
                      borderColor: 'primary.main',
                    },
                    transition: 'all 0.3s',
                  }}
                >
                  <Typography variant="body2" color="text.primary">
                    <strong>✨</strong> {feature}
                  </Typography>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>

      {/* Technical Details Accordion */}
      <Accordion sx={{ mt: 3 }}>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Typography variant="h6">🔬 Technical Details</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography variant="body2" paragraph>
            <strong>Training Strategy:</strong> The system uses a sliding window approach where 85% of historical
            data is used for training and 15% (most recent) is used for validation. This ensures the model
            learns from past patterns while being evaluated on recent, unseen data.
          </Typography>
          <Typography variant="body2" paragraph>
            <strong>Learning Mechanism:</strong> Given a sequence of prices from time t-10 to t, the model
            learns to predict the price at time t+1. This sequential learning captures temporal dependencies
            and market trends.
          </Typography>
          <Typography variant="body2" paragraph>
            <strong>Performance Tracking:</strong> Every model version is saved with comprehensive metrics
            (MAE, RMSE, MAPE). The system automatically compares new versions with previous ones to track
            improvement over time.
          </Typography>
          <Typography variant="body2" paragraph>
            <strong>Ensemble Intelligence:</strong> Multiple model types (LSTM, GRU, ARIMA) are trained
            independently. Their predictions are combined using performance-based weighting, where better
            performing models receive higher weight.
          </Typography>
          <Typography variant="body2">
            <strong>Continuous Improvement:</strong> As new market data becomes available, models are
            incrementally updated without losing previous learning. This allows the system to adapt to
            changing market conditions while retaining historical knowledge.
          </Typography>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default AdaptiveLearningExplainer;
