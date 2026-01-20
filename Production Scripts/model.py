"""
Neural Network Architecture for Sales Prediction
Optimized for Windows + Intel Ultra 7 255H (without IPEX)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class TemporalAttention(nn.Module):
    """Multi-scale temporal attention mechanism"""
    
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.week_attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.month_attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.quarter_attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.year_attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, week_feat, month_feat, quarter_feat, year_feat):
        week_out, _ = self.week_attn(week_feat, week_feat, week_feat)
        month_out, _ = self.month_attn(month_feat, month_feat, month_feat)
        quarter_out, _ = self.quarter_attn(quarter_feat, quarter_feat, quarter_feat)
        year_out, _ = self.year_attn(year_feat, year_feat, year_feat)
        
        combined = torch.cat([
            week_out[:, -1], month_out[:, -1],
            quarter_out[:, -1], year_out[:, -1]
        ], dim=-1)
        
        return self.fusion(combined)


class SalesPredictionModel(nn.Module):
    """
    Advanced sales prediction model with:
    - Sequential modeling (LSTM + Attention)
    - Multi-scale temporal analysis
    - Customer behavior encoding
    - Multi-task learning
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
        
        # Sequential encoder (LSTM)
        self.sequence_input_size = embed_dim + 3
        self.lstm = nn.LSTM(
            self.sequence_input_size,
            hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        
        # Temporal comparison encoders
        self.temporal_lstm_size = hidden_size // 2
        self.week_lstm = nn.LSTM(1, self.temporal_lstm_size, batch_first=True)
        self.month_lstm = nn.LSTM(1, self.temporal_lstm_size, batch_first=True)
        self.quarter_lstm = nn.LSTM(1, self.temporal_lstm_size, batch_first=True)
        self.year_lstm = nn.LSTM(1, self.temporal_lstm_size, batch_first=True)
        
        # Multi-scale attention
        self.temporal_attention = TemporalAttention(
            self.temporal_lstm_size, num_heads=4, dropout=dropout
        )
        
        # Main attention
        self.attention = nn.MultiheadAttention(
            hidden_size * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Customer feature encoder
        self.customer_encoder = nn.Sequential(
            nn.Linear(num_customer_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2)
        )
        
        # Feature fusion
        fusion_input_size = hidden_size * 2 + self.temporal_lstm_size + hidden_size // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        self.product_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_products)
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
        week_change: torch.Tensor,
        month_change: torch.Tensor,
        quarter_change: torch.Tensor,
        year_change: torch.Tensor,
        customer_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        # Product embeddings
        prod_embed = self.product_embedding(product_ids)
        
        # Combine sequence features
        seq_input = torch.cat([
            prod_embed,
            qty.unsqueeze(-1),
            revenue.unsqueeze(-1),
            discount.unsqueeze(-1)
        ], dim=-1)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(seq_input)
        
        # Attention over sequence
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        sequence_repr = attn_out[:, -1, :]
        
        # Temporal comparison encoding
        week_out, _ = self.week_lstm(week_change.unsqueeze(-1))
        month_out, _ = self.month_lstm(month_change.unsqueeze(-1))
        quarter_out, _ = self.quarter_lstm(quarter_change.unsqueeze(-1))
        year_out, _ = self.year_lstm(year_change.unsqueeze(-1))
        
        # Multi-scale temporal attention
        temporal_repr = self.temporal_attention(
            week_out, month_out, quarter_out, year_out
        )
        
        # Customer encoding
        customer_repr = self.customer_encoder(customer_features)
        
        # Fusion
        combined = torch.cat([sequence_repr, temporal_repr, customer_repr], dim=-1)
        fused = self.fusion(combined)
        
        # Multi-task predictions
        return {
            'product': self.product_head(fused),
            'quantity': self.quantity_head(fused),
            'revenue': self.revenue_head(fused),
            'discount': self.discount_head(fused)
        }
    
    def get_embeddings(self, product_ids: torch.Tensor) -> torch.Tensor:
        """Get product embeddings for analysis"""
        return self.product_embedding(product_ids)


def create_model(config, num_products: int) -> SalesPredictionModel:
    """
    Factory function to create model from config
    Optimized for Windows + Intel CPUs
    """
    model = SalesPredictionModel(
        num_products=num_products,
        embed_dim=config.EMBED_DIM,
        hidden_size=config.HIDDEN_SIZE,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created successfully")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    return model