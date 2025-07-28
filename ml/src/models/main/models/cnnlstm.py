import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels=10, hidden_dim=128, num_layers=4):
        super(CNNBlock, self).__init__()

        # Adjust channel numbers (limit maximum value)
        channel_sizes = [32, 64, 128, 256][:num_layers]

        # Large downsampling in the first convolutional layer
        self.initial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channel_sizes[0],
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(channel_sizes[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Subsequent convolutional layers
        layers = []
        for i in range(len(channel_sizes) - 1):
            layers.extend(
                [
                    nn.Conv2d(
                        channel_sizes[i],
                        channel_sizes[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channel_sizes[i + 1]),
                    nn.ReLU(inplace=True),
                ]
            )

        self.conv_layers = nn.Sequential(*layers)

        # Feature dimension reduction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channel_sizes[-1], hidden_dim)
        self.dropout = nn.Dropout(0.3)  # slightly reduce dropout rate

    def forward(self, x):
        # Initial downsampling
        x = self.initial_conv(x)  # reduce size to 1/4

        # Feature extraction
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class CNNLSTM(nn.Module):
    def __init__(self, hidden_dim=128, cnn_layers=4, dropout=0.5):
        super(CNNLSTM, self).__init__()

        # Fix in_channels of CNN blocks to 10
        self.cnn_blocks = nn.ModuleList(
            [
                CNNBlock(
                    in_channels=10,  # fix input channels to 10
                    hidden_dim=hidden_dim,
                    num_layers=cnn_layers,
                )
                for _ in range(4)
            ]
        )

        # LSTM layer (no changes)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # Replace with ClassificationHead
        self.classification_head = ClassificationHead(
            seq_len=32,  # LSTM output size
            emb_size=1,  # treat as 1D data
            mlp_hidden_size=32,
            dropout_rate=dropout,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, h_in):
        batch_size = x.size(0)

        # Use shared CNN blocks
        cnn_outputs = []
        for t in range(4):
            x_t = x[:, t]
            feat = self.cnn_blocks[t](x_t)
            cnn_outputs.append(feat)

        cnn_outputs = torch.stack(cnn_outputs, dim=1)
        lstm_out, _ = self.lstm(cnn_outputs)
        last_output = lstm_out[:, -1]  # (batch, 32)

        # Classification with ClassificationHead
        output = self.classification_head(last_output.unsqueeze(-1))
        output = self.softmax(output)

        return output, None

    def freeze_feature_extractor(self):
        """Freeze all layers except the classification head for second stage training"""
        # Freeze CNN blocks
        for block in self.cnn_blocks:
            for param in block.parameters():
                param.requires_grad = False

        # Freeze LSTM layer
        for param in self.lstm.parameters():
            param.requires_grad = False

        # Keep ClassificationHead trainable
        for param in self.classification_head.parameters():
            param.requires_grad = True

        # Freeze Softmax layer parameters if any
        for param in self.softmax.parameters():
            param.requires_grad = False


# Add ClassificationHead definition
class ClassificationHead(torch.nn.Sequential):
    def __init__(self, seq_len, emb_size, mlp_hidden_size, dropout_rate=0.5):
        super().__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(seq_len * emb_size, mlp_hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(mlp_hidden_size, 16),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(16, 4),
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x
