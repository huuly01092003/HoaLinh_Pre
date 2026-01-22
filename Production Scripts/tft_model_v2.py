"""
TFT Model - ANTI-OVERFITTING VERSION
Smaller + More Regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualNetwork(nn.Module):
    """GRN with dropout"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.gate = nn.Linear(output_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.elu(out)
        out = self.dropout(out)  # Dropout after activation
        out = self.fc2(out)
        out = self.dropout(out)  # Dropout again
        
        gate = self.sigmoid(self.gate(out))
        out = out * gate
        
        if self.skip is not None:
            x = self.skip(x)
        
        out = self.layer_norm(out + x)
        
        return out


class VariableSelectionNetwork(nn.Module):
    """VSN with dropout"""
    
    def __init__(self, input_dim, num_features, hidden_dim, dropout=0.3):
        super().__init__()
        
        self.num_features = num_features
        
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_features)
        ])
        
        self.attention = nn.Linear(hidden_dim * num_features, num_features)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features_list):
        processed = [grn(feat) for grn, feat in zip(self.feature_grns, features_list)]
        
        concat = torch.cat(processed, dim=-1)
        concat = self.dropout(concat)
        
        weights = self.softmax(self.attention(concat))
        
        weighted = torch.stack(processed, dim=1)
        weights = weights.unsqueeze(-1)
        
        output = (weighted * weights).sum(dim=1)
        
        return output, weights.squeeze(-1)


class TemporalFusionTransformer(nn.Module):
    """TFT - ANTI-OVERFITTING VERSION"""
    
    def __init__(
        self,
        product_vocab_size: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. Embeddings with dropout
        self.product_embedding = nn.Embedding(
            product_vocab_size, 
            hidden_dim, 
            padding_idx=0
        )
        self.embedding_dropout = nn.Dropout(dropout)
        
        # 2. Encoders
        self.total_qty_encoder = nn.Linear(1, hidden_dim)
        self.num_skus_encoder = nn.Linear(1, hidden_dim)
        
        # 3. VSN
        self.vsn = VariableSelectionNetwork(
            input_dim=hidden_dim,
            num_features=2,  # products + total_qty (bỏ num_skus để giảm complexity)
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # 4. LSTM with dropout
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=1,  # REDUCED to 1 layer
            batch_first=True,
            dropout=0  # No LSTM dropout (use manual dropout)
        )
        self.lstm_dropout = nn.Dropout(dropout)
        
        # 5. Self-Attention
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 6. GRN layers
        self.enrichment = GatedResidualNetwork(
            hidden_dim, 
            hidden_dim * 2,  # REDUCED: *4 → *2
            hidden_dim, 
            dropout
        )
        self.temporal_fusion = GatedResidualNetwork(
            hidden_dim * 2, 
            hidden_dim * 2,  # REDUCED: *4 → *2
            hidden_dim, 
            dropout
        )
        
        # 7. Prediction heads with Label Smoothing
        self.product_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, product_vocab_size)
        )
        
        self.num_skus_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.ReLU()
        )
        
        self.total_qty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.ReLU()
        )
        
        self.days_until_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.ReLU()
        )
    
    def encode_history(self, hist_products, hist_quantities, hist_total_qty, hist_num_skus, hist_days_between):
        """Encode history with dropout"""
        batch_size, history_len = hist_total_qty.shape
        
        # Embed products with dropout
        product_embeds = self.product_embedding(hist_products)
        product_embeds = self.embedding_dropout(product_embeds)
        product_embeds = product_embeds.mean(dim=2)
        
        # Encode total quantity
        total_qty_embeds = self.total_qty_encoder(hist_total_qty.unsqueeze(-1))
        
        # VSN (chỉ 2 features)
        selected_features = []
        
        for t in range(history_len):
            features_t = [
                product_embeds[:, t, :],
                total_qty_embeds[:, t, :]
            ]
            
            selected_t, _ = self.vsn(features_t)
            selected_features.append(selected_t)
        
        selected_seq = torch.stack(selected_features, dim=1)
        
        return selected_seq
    
    def forward(self, hist_products, hist_quantities, hist_total_qty, hist_num_skus, hist_days_between):
        """Forward with regularization"""
        
        # 1. Encode history
        encoded_history = self.encode_history(
            hist_products, hist_quantities, hist_total_qty, hist_num_skus, hist_days_between
        )
        
        # 2. LSTM with dropout
        lstm_out, _ = self.lstm(encoded_history)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # 3. Self-attention
        attn_out, _ = self.self_attention(lstm_out, lstm_out, lstm_out)
        
        # 4. Enrichment
        enriched = self.enrichment(attn_out)
        
        # 5. Temporal fusion
        combined = torch.cat([lstm_out, enriched], dim=-1)
        fused = self.temporal_fusion(combined)
        
        # 6. Take last timestep
        final_repr = fused[:, -1, :]
        
        # 7. Predictions
        predictions = {
            'products': self.product_head(final_repr),
            'num_skus': self.num_skus_head(final_repr),
            'total_qty': self.total_qty_head(final_repr),
            'days_until_next': self.days_until_head(final_repr)
        }
        
        return predictions


def create_tft_model(config, vocab_size):
    """Factory function"""
    model = TemporalFusionTransformer(
        product_vocab_size=vocab_size,
        hidden_dim=config.HIDDEN_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print("\n" + "="*80)
    print("TFT MODEL - ANTI-OVERFITTING VERSION")
    print("="*80)
    print(f"Focus: Products (NO MONEY)")
    print(f"Regularization: HIGH (dropout={config.DROPOUT})")
    print(f"Vocabulary: {vocab_size:,}")
    print(f"Hidden dim: {config.HIDDEN_DIM} (REDUCED)")
    print(f"Layers: {config.NUM_LAYERS} (REDUCED)")
    print(f"Parameters: {total_params:,}")
    print(f"Size: {total_params * 4 / 1024 / 1024:.2f} MB")
    print("="*80 + "\n")
    
    return model


if __name__ == "__main__":
    from config_v2 import SystemConfig
    
    model = create_tft_model(SystemConfig, vocab_size=245)
    
    batch_size = 16
    history_len = 15
    max_products = 10
    
    dummy_input = {
        'hist_products': torch.randint(0, 245, (batch_size, history_len, max_products)),
        'hist_quantities': torch.randn(batch_size, history_len, max_products).abs(),
        'hist_total_qty': torch.randn(batch_size, history_len),
        'hist_num_skus': torch.randn(batch_size, history_len),
        'hist_days_between': torch.randn(batch_size, history_len)
    }
    
    with torch.no_grad():
        outputs = model(**dummy_input)
    
    print("✅ Forward pass:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")