"""
Time-Series Aware Sales Forecasting Model
Predicts customer behavior for a SPECIFIC future date
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import math


class TimeEmbedding(nn.Module):
    """Sinusoidal time encoding for temporal awareness"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, time_features: torch.Tensor):
        """
        Args:
            time_features: [batch, 4] - [day_of_week, day_of_month, month, days_until_prediction]
        Returns:
            [batch, d_model] time embeddings
        """
        batch_size = time_features.size(0)
        
        # Separate features
        day_of_week = time_features[:, 0]  # 0-6
        day_of_month = time_features[:, 1]  # 1-31
        month = time_features[:, 2]  # 1-12
        days_until = time_features[:, 3]  # forecast horizon
        
        # Sinusoidal encoding for each temporal feature
        def encode(values, max_val, dim):
            # Normalize to [0, 1]
            normalized = values / max_val
            # Create position encodings
            pe = torch.zeros(batch_size, dim // 4, device=values.device)
            position = normalized.unsqueeze(1)
            
            div_term = torch.exp(torch.arange(0, dim // 4, 2, device=values.device).float() * 
                                 (-math.log(10000.0) / (dim // 4)))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe
        
        # Encode each temporal dimension
        dow_enc = encode(day_of_week, 7, self.d_model)
        dom_enc = encode(day_of_month, 31, self.d_model)
        month_enc = encode(month, 12, self.d_model)
        horizon_enc = encode(days_until, 30, self.d_model)
        
        # Combine
        time_emb = dow_enc + dom_enc + month_enc + horizon_enc
        return time_emb


class TimeSeriesSalesModel(nn.Module):
    """
    Time-aware sales forecasting model
    Predicts what customer will buy on a SPECIFIC future date
    """
    
    def __init__(
        self,
        num_products: int,
        embed_dim: int = 128,
        hidden_size: int = 256,
        num_heads: int = 8,
        dropout: float = 0.3,
        num_customer_features: int = 14
    ):
        super().__init__()
        
        self.num_products = num_products
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        
        # Product embedding
        self.product_embedding = nn.Embedding(
            num_products + 1, embed_dim, padding_idx=0
        )
        
        # Time embedding
        self.time_embedding = TimeEmbedding(embed_dim)
        
        # Historical sequence encoder (LSTM)
        self.sequence_input_size = embed_dim + 3  # product + qty + revenue + discount
        self.lstm = nn.LSTM(
            self.sequence_input_size,
            hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        
        # Temporal pattern encoder
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size * 2,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Customer feature encoder
        self.customer_encoder = nn.Sequential(
            nn.Linear(num_customer_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2)
        )
        
        # Time-aware fusion
        fusion_input_size = hidden_size * 2 + embed_dim + hidden_size // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-task prediction heads
        self.product_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_products)
        )
        
        # Probability of purchase (will customer buy or not?)
        self.purchase_probability_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.quantity_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.ReLU()
        )
        
        self.revenue_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.ReLU()
        )
        
        self.discount_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        product_ids: torch.Tensor,
        qty: torch.Tensor,
        revenue: torch.Tensor,
        discount: torch.Tensor,
        customer_features: torch.Tensor,
        target_time_features: torch.Tensor  # NEW: [day_of_week, day_of_month, month, days_until]
    ) -> Dict[str, torch.Tensor]:
        
        # Product embeddings from history
        prod_embed = self.product_embedding(product_ids)
        
        # Combine sequence features
        seq_input = torch.cat([
            prod_embed,
            qty.unsqueeze(-1),
            revenue.unsqueeze(-1),
            discount.unsqueeze(-1)
        ], dim=-1)
        
        # Encode historical sequence
        lstm_out, _ = self.lstm(seq_input)
        
        # Apply transformer for temporal patterns
        temporal_out = self.temporal_encoder(lstm_out)
        sequence_repr = temporal_out[:, -1, :]  # Take last timestep
        
        # Encode target time (future date we're predicting for)
        time_emb = self.time_embedding(target_time_features)
        
        # Customer encoding
        customer_repr = self.customer_encoder(customer_features)
        
        # Fusion: history + future time + customer
        combined = torch.cat([sequence_repr, time_emb, customer_repr], dim=-1)
        fused = self.fusion(combined)
        
        # Multi-task predictions
        return {
            'product': self.product_head(fused),
            'purchase_probability': self.purchase_probability_head(fused),
            'quantity': self.quantity_head(fused),
            'revenue': self.revenue_head(fused),
            'discount': self.discount_head(fused)
        }


def create_timeseries_model(config, num_products: int) -> TimeSeriesSalesModel:
    """Factory function for time-series model"""
    model = TimeSeriesSalesModel(
        num_products=num_products,
        embed_dim=config.EMBED_DIM,
        hidden_size=config.HIDDEN_SIZE,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Time-Series Model created")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    return model