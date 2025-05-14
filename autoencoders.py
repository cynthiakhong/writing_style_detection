import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, dropout_rate=0.2):
        """
        Autoencoder for dimensionality reduction, optimized for text
        
        Args:
            input_dim: Dimension of input embeddings
            encoding_dim: Dimension of the latent space
            dropout_rate: Dropout probability
        """
        super(Autoencoder, self).__init__()
        
        hidden_1 = min(768, input_dim)
        hidden_2 = min(384, hidden_1 // 2)
        hidden_3 = min(192, hidden_2 // 2)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_1, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate/2),
            
            nn.Linear(hidden_2, hidden_3),
            nn.LayerNorm(hidden_3),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_3, encoding_dim),
            nn.Tanh()  # Bounded activation for latent space
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_3),
            nn.LayerNorm(hidden_3),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_3, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate/2),
            
            nn.Linear(hidden_2, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_1, input_dim)
        )
        
        # Initialize weights with Xavier/Glorot
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
     
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, z):
        return self.decoder(z)


class VAE(nn.Module):
    def __init__(self, input_dim, encoding_dim=64, hidden_dim=128, dropout_rate=0.2):
        """
        Variational Autoencoder for dimensionality reduction, optimized for text
        
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layers
            encoding_dim: Dimension of the latent space
            dropout_rate: Dropout probability
        """
        super(VAE, self).__init__()
        
        # More adaptive sizing for different embedding dimensions
        hidden_1 = min(hidden_dim*2, input_dim)
        hidden_2 = min(hidden_dim, input_dim//2)
        
        # Encoder layers - keeping your structure but adjusting activations
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.GELU(),  # Changed from LeakyReLU to GELU for better transformer compatibility
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_1, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.GELU(),  # Changed to GELU
            nn.Dropout(dropout_rate/2),  # Graduated dropout
        )
        
        # Mean and log variance layers for the latent space
        self.fc_mu = nn.Linear(hidden_2, encoding_dim)
        self.fc_logvar = nn.Linear(hidden_2, encoding_dim)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.GELU(),  # Changed to GELU
            nn.Dropout(dropout_rate/2),  # Graduated dropout
            
            nn.Linear(hidden_2, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.GELU(),  # Changed to GELU
            nn.Dropout(dropout_rate/4),  # Lower dropout near output
            
            nn.Linear(hidden_1, input_dim)
        )
                
        # Initialize weights with more modern initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Slightly better initialization for GELU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from latent space"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """Decode latent samples back to input space"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through the VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def get_latent(self, x):
        """Get the latent representation without sampling"""
        mu, _ = self.encode(x)
        return mu
    
    @staticmethod
    def kl_loss(mu, logvar):
        """
        Calculate KL divergence loss with clamping for stability
        Can be called statically: VAE.kl_loss(mu, logvar)
        """
        # KL divergence with clamping for numerical stability
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        # Clamp extremely large values that might cause instability
        kld = torch.clamp(kld, min=0.0, max=50.0)
        return kld.mean()
    
    @staticmethod
    def reconstruction_loss(x_reconstructed, x_original, reduction='mean'):
        """
        Calculate reconstruction loss (MSE with Huber-like properties)
        Can be called statically: VAE.reconstruction_loss(x_reconstructed, x_original)
        """
        # Use a combination of MSE and smoother L1 loss for better handling of outliers
        mse_loss = F.mse_loss(x_reconstructed, x_original, reduction='none')
        smooth_l1_loss = F.smooth_l1_loss(x_reconstructed, x_original, reduction='none', beta=0.1)
        
        # Blend the losses - majority MSE with some robustness from smooth L1
        combined_loss = 0.9 * mse_loss + 0.1 * smooth_l1_loss
        
        if reduction == 'mean':
            return combined_loss.mean()
        elif reduction == 'sum':
            return combined_loss.sum()
        else:
            return combined_loss
    
    @staticmethod
    def vae_loss(x_reconstructed, x_original, mu, logvar, kl_weight=0.5):
        """
        Calculate total VAE loss with balanced KL weight
        Can be called statically: VAE.vae_loss(x_reconstructed, x_original, mu, logvar)
        """
        recon_loss = VAE.reconstruction_loss(x_reconstructed, x_original)
        kl_loss = VAE.kl_loss(mu, logvar)
        
        # Lower KL weight to focus more on reconstruction quality
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss