# 🚀 Sales Prediction System - Hybrid Production & Experimentation

A comprehensive deep learning system for sales prediction with support for both production deployment and interactive experimentation.

## 📁 Project Structure

```
sales-prediction/
├── config.py                 # Configuration management
├── geocache.py              # Geographic caching (34 provinces)
├── features.py              # Feature engineering pipeline
├── model.py                 # Neural network architecture
├── dataset.py               # PyTorch dataset classes
├── train.py                 # Training and evaluation
├── main.py                  # Production training script
├── experiment.ipynb         # Jupyter notebook for experiments
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── data/
│   ├── raw/                # Raw CSV files
│   └── processed/          # Processed features
│
├── models/                 # Saved models and checkpoints
│   ├── best_model.pth
│   ├── checkpoints/
│   └── encoders.pkl
│
├── logs/                   # Training logs
│   └── training.log
│
└── cache/                  # Geocoding cache
    └── geocache.pkl
```

## 🎯 Key Features

### Production-Ready Components (`.py` files)

- **Modular Architecture**: Clean separation of concerns
- **Configuration Management**: Environment-specific configs
- **Robust Feature Engineering**: Automated pipeline with geocoding
- **Multi-task Learning**: Product, quantity, revenue, discount prediction
- **Temporal Modeling**: Multi-scale temporal attention (week/month/quarter/year)
- **Geographic Intelligence**: Vietnam 34-province structure (2025)
- **Early Stopping**: Automatic checkpointing
- **Logging**: Comprehensive logging system

### Experimentation Tools (`.ipynb` notebook)

- **Interactive Exploration**: Step-by-step data analysis
- **Visualization**: Rich charts and plots
- **Rapid Prototyping**: Quick hyperparameter tuning
- **Feature Analysis**: Correlation, importance, embeddings
- **Model Interpretation**: Predictions, residuals, confusion matrix
- **Experiment Tracking**: Automatic result saving

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd sales-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Place your data file
mkdir -p data/raw
cp merged_2025.csv data/raw/
```

### 3. Production Training

```bash
# Development mode (faster, more logging)
python main.py --env development

# Production mode (optimized)
python main.py --env production

# Experiment mode (for research)
python main.py --env experiment
```

### 4. Interactive Experimentation

```bash
# Launch Jupyter
jupyter notebook experiment.ipynb

# Or use JupyterLab
jupyter lab
```

## 📊 Model Architecture

```
Input Sequences (Products, Quantities, Revenues, Discounts)
    ↓
Product Embeddings (128-dim)
    ↓
Bidirectional LSTM (256 hidden, 2 layers)
    ↓
Multi-head Attention (8 heads)
    ↓
Temporal Comparison LSTMs (Week/Month/Quarter/Year)
    ↓
Multi-scale Temporal Attention
    ↓
Customer Feature Encoder
    ↓
Feature Fusion Layer
    ↓
Multi-task Heads:
    ├── Product Classification (CrossEntropy)
    ├── Quantity Regression (SmoothL1)
    ├── Revenue Regression (SmoothL1)
    └── Discount Regression (SmoothL1 + Sigmoid)
```

## 🔧 Configuration

### Environment Variables

Edit `config.py` to customize:

```python
class Config:
    BATCH_SIZE = 128
    EPOCHS = 20
    LEARNING_RATE = 0.001
    EMBED_DIM = 128
    HIDDEN_SIZE = 256
    NUM_HEADS = 8
    DROPOUT = 0.3
    MAX_SEQ_LEN = 30
```

### Production vs Development

```python
# Production: Optimized for speed
config = ProductionConfig()
config.BATCH_SIZE = 256
config.NUM_WORKERS = 8

# Development: Fast iteration
config = DevelopmentConfig()
config.BATCH_SIZE = 64
config.NUM_WORKERS = 2
```

## 📈 Training Pipeline

### 1. Data Loading
- Load CSV with low_memory=False
- Train/Val/Test split (85/10/5)

### 2. Feature Engineering
- **Temporal Features**: Year, month, quarter, week, hour
- **Product Features**: Series, family, price tier
- **Customer Features**: RFM analysis, segments
- **Geographic Features**: Province, district, coordinates, distances
- **Aggregated Features**: Customer lifetime value, product diversity
- **Temporal Comparisons**: Week-over-week, month-over-month, etc.

### 3. Geocoding
- Cache-based system (34 provinces)
- Haversine distance calculations
- Urban/rural classification

### 4. Sequence Creation
- Customer-level purchase sequences
- Padding to MAX_SEQ_LEN
- Multi-scale temporal features

### 5. Model Training
- AdamW optimizer with weight decay
- Cosine annealing with warm restarts
- Gradient clipping
- Early stopping with patience

### 6. Evaluation
- Product prediction accuracy
- Revenue MAE and RMSE
- Quantity MAE
- Discount MAE

## 📊 Results Interpretation

### Metrics Explained

```python
{
    'product_accuracy': 0.75,      # 75% correct product predictions
    'revenue_mae': 50000,          # Avg error: 50,000 VND
    'revenue_rmse': 75000,         # Root mean squared error
    'quantity_mae': 2.5,           # Avg quantity error: 2.5 units
    'discount_mae': 0.05           # Avg discount error: 5%
}
```

### Typical Performance

- **Product Accuracy**: 70-80%
- **Revenue MAE**: 40,000-60,000 VND
- **Training Time**: 30-60 minutes (20 epochs, GPU)

## 🔬 Experimentation Workflow

### 1. Data Exploration (Cells 3-8)
```python
# Load and visualize data
df = pd.read_csv(config.RAW_DATA_PATH)
df.describe()

# Geographic distribution
province_dist.plot(kind='bar')

# Temporal trends
temporal_dist.plot()
```

### 2. Feature Engineering (Cells 9-12)
```python
# Process features
engineer = FeatureEngineer(train_df, config.CURRENT_DATE, geo_cache)
engineer.process(fit_mode=True)

# Create sequences
train_sequences = engineer.prepare_sequences(config.MAX_SEQ_LEN)
```

### 3. Model Training (Cells 13-16)
```python
# Initialize and train
model = create_model(config, num_products)
trainer = Trainer(model, train_loader, val_loader, config)
history = trainer.train(epochs=config.EPOCHS)
```

### 4. Analysis (Cells 17-22)
```python
# Evaluate
test_metrics, preds, targets = evaluate_model(model, test_loader, device)

# Visualize
plt.scatter(targets, preds)
plt.plot(history['val_loss'])
```

## 🎓 Advanced Usage

### Custom Features

```python
# Add your own features in features.py
class FeatureEngineer:
    def create_custom_features(self):
        # Your feature logic
        self.df['my_feature'] = ...
        return self
```

### Hyperparameter Tuning

```python
# In experiment.ipynb
configs_to_try = [
    {'LEARNING_RATE': 0.001, 'HIDDEN_SIZE': 128},
    {'LEARNING_RATE': 0.0005, 'HIDDEN_SIZE': 256},
    {'LEARNING_RATE': 0.0001, 'HIDDEN_SIZE': 512}
]

for params in configs_to_try:
    config.update(params)
    # Train and evaluate
    ...
```

### Model Ensemble

```python
# Train multiple models
models = []
for seed in [42, 123, 456]:
    config.RANDOM_STATE = seed
    model = create_model(config, num_products)
    # Train model
    models.append(model)

# Average predictions
ensemble_pred = np.mean([m.predict(X) for m in models], axis=0)
```

## 🐛 Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
config.BATCH_SIZE = 32

# Reduce sequence length
config.MAX_SEQ_LEN = 20

# Use gradient accumulation
accumulation_steps = 4
```

### Slow Training

```python
# Increase num_workers
config.NUM_WORKERS = 8

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Poor Convergence

```python
# Lower learning rate
config.LEARNING_RATE = 0.0001

# Increase patience
config.PATIENCE = 10

# Add more regularization
config.DROPOUT = 0.5
```

## 📦 Deployment

### Save for Production

```python
# Export model
torch.save({
    'model_state_dict': model.state_dict(),
    'product_to_id': product_to_id,
    'config': config.to_dict()
}, 'production_model.pth')
```

### Inference API

```python
# Load model
checkpoint = torch.load('production_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
def predict(customer_sequence):
    with torch.no_grad():
        outputs = model(customer_sequence)
    return outputs
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Vietnam 34-province structure based on Nghị quyết 202/2025/QH15
- Geocoding system optimized for Vietnamese addresses
- Model architecture inspired by modern sequence modeling research

## 📧 Contact

For questions or support, please open an issue or contact [your-email@example.com]

---

**Happy Predicting! 🎯📊🚀**