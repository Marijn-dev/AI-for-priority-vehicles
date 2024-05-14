import torch
import torch.nn as nn
from encoder_transformer import Encoder
from decoder_transformer import Decoder

class TrajectoryAttentionModel(nn.Module):
    def __init__(
        self, 
        embed_size, 
        num_layers, 
        heads, 
        forward_expansion, 
        dropout, 
        device, 
        max_length, 
        input_dim=2
    ):
        super(TrajectoryAttentionModel, self).__init__()
        self.device = device

        self.word_embedding = nn.Linear(input_dim, embed_size)  # Input embedding layer
        self.position_embedding = nn.Parameter(torch.rand(max_length, embed_size))

        self.encoder = Encoder(
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout
        )

        self.decoder = Decoder(
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout
        )

        self.fc_out = nn.Linear(embed_size, input_dim)  # Output layer to predict coordinates
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, num_pred_steps):
        batch_size, seq_length, _ = src.size()
        src = self.word_embedding(src) + self.position_embedding[:seq_length]

        enc_src = self.encoder(src)

        outputs = enc_src[:, -1:, :].repeat(1, 1, 1)  # Initialize the first input for the decoder
        predictions = []

        for _ in range(num_pred_steps):
            output = self.decoder(outputs, enc_src)
            outputs = output[:, -1:, :]  # Only use the last output as the next input
            predictions.append(outputs)  # Store predictions

        predictions = torch.cat(predictions, dim=1)  # [Batch, num_pred_steps, Embed_Size]
        return self.fc_out(predictions)  # Map to the coordinate space



# # Example Usage
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Transformer(
#     embed_size=512, 
#     num_layers=6, 
#     heads=8, 
#     forward_expansion=4, 
#     dropout=0.1, 
#     device=device, 
#     max_length=100
# ).to(device)

# # Dummy data
# src = torch.rand(10, 30, 2).to(device)  # Batch size = 10, Sequence length = 30
# pred_steps = 5  # Predict next 5 steps

# predictions = model(src, pred_steps)
# print(predictions.shape)  # Expected shape: [10, 5, 2]
