import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
# import dgl
# from dgl.nn import GATConv
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src.dlutils import *
from src.constants import *
# torch.manual_seed(1)

# new for iTransformer
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

## Separate LSTM for each variable
class LSTM_Univariate(nn.Module):
	def __init__(self, feats):
		super(LSTM_Univariate, self).__init__()
		self.name = 'LSTM_Univariate'
		self.lr = 0.002
		self.n_feats = feats
		self.n_hidden = 1
		self.lstm = nn.ModuleList([nn.LSTM(1, self.n_hidden) for i in range(feats)])

	def forward(self, x):
		hidden = [(torch.rand(1, 1, self.n_hidden, dtype=torch.float64), 
			torch.randn(1, 1, self.n_hidden, dtype=torch.float64)) for i in range(self.n_feats)]
		outputs = []
		for i, g in enumerate(x):
			multivariate_output = []
			for j in range(self.n_feats):
				univariate_input = g.view(-1)[j].view(1, 1, -1)
				out, hidden[j] = self.lstm[j](univariate_input, hidden[j])
				multivariate_output.append(2 * out.view(-1))
			output = torch.cat(multivariate_output)
			outputs.append(output)
		return torch.stack(outputs)

## Simple Multi-Head Self-Attention Model
class Attention(nn.Module):
	def __init__(self, feats):
		super(Attention, self).__init__()
		self.name = 'Attention'
		self.lr = 0.0001
		self.n_feats = feats
		self.window_size = 5 # MHA w_size = 5
		self.n = self.n_feats * self.window_size
		self.atts = [ nn.Sequential( nn.Linear(self.n, feats * feats), 
				nn.ReLU(True))	for i in range(1)]
		self.atts = nn.ModuleList(self.atts)

	def forward(self, g):
		for at in self.atts:
			ats = at(g.view(-1)).reshape(self.n_feats, self.n_feats)
			g = torch.matmul(g, ats)		
		return g, ats

## LSTM_AD Model
class LSTM_AD(nn.Module):
	def __init__(self, feats, window_size=None, prob=False):
		super(LSTM_AD, self).__init__()
		self.name = 'LSTM_AD'
		self.lr = 0.002
		self.n_feats = feats
		self.n_hidden = 64
		self.lstm = nn.LSTM(feats, self.n_hidden)
		self.lstm2 = nn.LSTM(feats, self.n_feats)
		self.fcn = nn.Sequential(nn.Linear(self.n_feats, self.n_feats), nn.Sigmoid())

	def forward(self, x):
		hidden = (torch.rand(1, 1, self.n_hidden, dtype=torch.float64), torch.randn(1, 1, self.n_hidden, dtype=torch.float64))
		hidden2 = (torch.rand(1, 1, self.n_feats, dtype=torch.float64), torch.randn(1, 1, self.n_feats, dtype=torch.float64))
		outputs = []
		for i, g in enumerate(x):
			out, hidden = self.lstm(g.view(1, 1, -1), hidden)
			out, hidden2 = self.lstm2(g.view(1, 1, -1), hidden2)
			out = self.fcn(out.view(-1))
			outputs.append(2 * out.view(-1))
		return torch.stack(outputs)
	
## LSTM_AE Model (as done for Vilius' QT)
class LSTM_AE(nn.Module):
	def __init__(self, feats, window_size=None, prob=False):
		super(LSTM_AE, self).__init__()
		self.name = 'LSTM_AE'
		self.lr = 0.002
		self.batch = 100
		self.n_feats = feats
		self.n_hidden = 64
		self.window_size = window_size
		self.lstm = nn.LSTM(input_size=self.window_size, hidden_size=self.n_hidden, num_layers=1, batch_first=True)
		self.lstm2 = nn.LSTM(input_size=self.n_hidden, hidden_size=self.n_hidden, num_layers=1, batch_first=True)
		self.fcn = nn.Linear(self.n_hidden, self.n_feats)

	def forward(self, x):	
		x = x.permute(0, 2, 1)
		# outputs = torch.tensor([])
		# for g in x:
		# 	g.unsqueeze_(0)
		# 	out1, (h1, c1) = self.lstm(g)
		# 	latent = h1.repeat(self.window_size, 1, 1)
		# 	latent = latent.swapdims(0, 1)
		# 	out2, (h2, c2) = self.lstm2(latent)
		# 	out2 = self.fcn(out2)
		# 	# outputs.append(out2)
		# 	outputs = torch.cat([outputs, out2], dim=0)
		# x.unsqueeze_(0)
		out1, (h1, c1) = self.lstm(x)
		latent = h1.repeat(self.window_size, 1, 1)
		latent = latent.swapdims(0, 1)
		out2, (h2, c2) = self.lstm2(latent)
		out2 = self.fcn(out2)
		outputs = out2
		# outputs.append(out2)
		# outputs = torch.cat([outputs, out2], dim=0)
		# outputs = torch.stack(outputs)
		# outputs = torch.cat(outputs)
		# outputs = outputs.view(-1, self.window_size, self.n_feats)
		# outputs = torch.squeeze(torch.stack(outputs))
		return outputs
	
## DAGMM Model (ICLR 18)
class DAGMM(nn.Module):
	def __init__(self, feats, window_size=5, prob=False):
		super(DAGMM, self).__init__()
		self.name = 'DAGMM'
		self.lr = 0.0001
		self.beta = 0.01
		self.n_feats = feats
		self.n_hidden = 16
		self.n_latent = 8
		self.window_size = window_size # DAGMM w_size = 5
		self.n = self.n_feats * self.window_size
		self.n_gmm = self.n_feats * self.window_size
		self.encoder = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.estimate = nn.Sequential(
			nn.Linear(self.n_latent+2, self.n_hidden), nn.Tanh(), nn.Dropout(0.5),
			nn.Linear(self.n_hidden, self.n_gmm), nn.Softmax(dim=1),
		)

	def compute_reconstruction(self, x, x_hat):
		relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / x.norm(2, dim=1)
		cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
		return relative_euclidean_distance, cosine_similarity

	def forward(self, x):
		## Encode Decoder
		x = x.view(1, -1)
		z_c = self.encoder(x)
		x_hat = self.decoder(z_c)
		## Compute Reconstructoin
		rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
		z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
		## Estimate
		gamma = self.estimate(z)
		return z_c, x_hat.view(-1), z, gamma.view(-1)

## OmniAnomaly Model (KDD 19)
class OmniAnomaly(nn.Module):
	def __init__(self, feats, window_size=None, prob=False):
		super(OmniAnomaly, self).__init__()
		self.name = 'OmniAnomaly'
		self.lr = 0.002
		self.beta = 0.01
		self.n_feats = feats
		self.n_hidden = 32
		self.n_latent = 8
		self.lstm = nn.GRU(feats, self.n_hidden, 2)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Flatten(),
			nn.Linear(self.n_hidden, 2*self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
		)

	def forward(self, x, hidden = None):
		hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
		out, hidden = self.lstm(x.view(1, 1, -1), hidden)
		## Encode
		x = self.encoder(out)
		mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
		## Reparameterization trick
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		x = mu + eps*std
		## Decoder
		x = self.decoder(x)
		return x.view(-1), mu.view(-1), logvar.view(-1), hidden

## USAD Model (KDD 20)
class USAD(nn.Module):
	def __init__(self, feats,  window_size=None, prob=False):
		super(USAD, self).__init__()
		self.name = 'USAD'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_hidden = 16
		self.n_latent = 5
		self.window_size = 5 # USAD w_size = 5
		self.n = self.n_feats * self.window_size
		self.encoder = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
		)
		self.decoder1 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.decoder2 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)

	def forward(self, g):
		## Encode
		z = self.encoder(g.view(1,-1))
		## Decoders (Phase 1)
		ae1 = self.decoder1(z)
		ae2 = self.decoder2(z)
		## Encode-Decode (Phase 2)
		ae2ae1 = self.decoder2(self.encoder(ae1))
		return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)

## MSCRED Model (AAAI 19)
class MSCRED(nn.Module):
	def __init__(self, feats):
		super(MSCRED, self).__init__()
		self.name = 'MSCRED'
		self.lr = 0.0001
		self.n_feats = feats
		self.window_size = feats
		self.encoder = nn.ModuleList([
			ConvLSTM(1, 32, (3, 3), 1, True, True, False),
			ConvLSTM(32, 64, (3, 3), 1, True, True, False),
			ConvLSTM(64, 128, (3, 3), 1, True, True, False),
			]
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(128, 64, (3, 3), 1, 1), nn.ReLU(True),
			nn.ConvTranspose2d(64, 32, (3, 3), 1, 1), nn.ReLU(True),
			nn.ConvTranspose2d(32, 1, (3, 3), 1, 1), nn.Sigmoid(),
		)

	def forward(self, g):
		## Encode
		z = g.view(1, 1, self.n_feats, self.window_size)
		for cell in self.encoder:
			_, z = cell(z.view(1, *z.shape))
			z = z[0][0]
		## Decode
		x = self.decoder(z)
		return x.view(-1)

## CAE-M Model (TKDE 21)
class CAE_M(nn.Module):
	def __init__(self, feats):
		super(CAE_M, self).__init__()
		self.name = 'CAE_M'
		self.lr = 0.001
		self.n_feats = feats
		self.window_size = feats
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 8, (3, 3), 1, 1), nn.Sigmoid(),
			nn.Conv2d(8, 16, (3, 3), 1, 1), nn.Sigmoid(),
			nn.Conv2d(16, 32, (3, 3), 1, 1), nn.Sigmoid(),
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(32, 4, (3, 3), 1, 1), nn.Sigmoid(),
			nn.ConvTranspose2d(4, 4, (3, 3), 1, 1), nn.Sigmoid(),
			nn.ConvTranspose2d(4, 1, (3, 3), 1, 1), nn.Sigmoid(),
		)

	def forward(self, g):
		## Encode
		z = g.view(1, 1, self.n_feats, self.window_size)
		z = self.encoder(z)
		## Decode
		x = self.decoder(z)
		return x.view(-1)

# ## MTAD_GAT Model (ICDM 20)
# class MTAD_GAT(nn.Module):
# 	def __init__(self, feats):
# 		super(MTAD_GAT, self).__init__()
# 		self.name = 'MTAD_GAT'
# 		self.lr = 0.0001
# 		self.n_feats = feats
# 		self.window_size = feats
# 		self.n_hidden = feats * feats
# 		self.g = dgl.graph((torch.tensor(list(range(1, feats+1))), torch.tensor([0]*feats)))
# 		self.g = dgl.add_self_loop(self.g)
# 		self.feature_gat = GATConv(feats, 1, feats)
# 		self.time_gat = GATConv(feats, 1, feats)
# 		self.gru = nn.GRU((feats+1)*feats*3, feats*feats, 1)

# 	def forward(self, data, hidden):
# 		hidden = torch.rand(1, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
# 		data = data.view(self.window_size, self.n_feats)
# 		data_r = torch.cat((torch.zeros(1, self.n_feats), data))
# 		feat_r = self.feature_gat(self.g, data_r)
# 		data_t = torch.cat((torch.zeros(1, self.n_feats), data.t()))
# 		time_r = self.time_gat(self.g, data_t)
# 		data = torch.cat((torch.zeros(1, self.n_feats), data))
# 		data = data.view(self.window_size+1, self.n_feats, 1)
# 		x = torch.cat((data, feat_r, time_r), dim=2).view(1, 1, -1)
# 		x, h = self.gru(x, hidden)
# 		return x.view(-1), h

# ## GDN Model (AAAI 21)
# class GDN(nn.Module):
# 	def __init__(self, feats):
# 		super(GDN, self).__init__()
# 		self.name = 'GDN'
# 		self.lr = 0.0001
# 		self.n_feats = feats
# 		self.window_size = 5
# 		self.n_hidden = 16
# 		self.n = self.window_size * self.n_feats
# 		src_ids = np.repeat(np.array(list(range(feats))), feats)
# 		dst_ids = np.array(list(range(feats))*feats)
# 		self.g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)))
# 		self.g = dgl.add_self_loop(self.g)
# 		self.feature_gat = GATConv(1, 1, feats)
# 		self.attention = nn.Sequential(
# 			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
# 			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
# 			nn.Linear(self.n_hidden, self.window_size), nn.Softmax(dim=0),
# 		)
# 		self.fcn = nn.Sequential(
# 			nn.Linear(self.n_feats, self.n_hidden), nn.LeakyReLU(True),
# 			nn.Linear(self.n_hidden, self.window_size), nn.Sigmoid(),
# 		)

	# def forward(self, data):
	# 	# Bahdanau style attention
	# 	att_score = self.attention(data).view(self.window_size, 1)
	# 	data = data.view(self.window_size, self.n_feats)
	# 	data_r = torch.matmul(data.permute(1, 0), att_score)
	# 	# GAT convolution on complete graph
	# 	feat_r = self.feature_gat(self.g, data_r)
	# 	feat_r = feat_r.view(self.n_feats, self.n_feats)
	# 	# Pass through a FCN
	# 	x = self.fcn(feat_r)
	# 	return x.view(-1)

# MAD_GAN (ICANN 19)
class MAD_GAN(nn.Module):
	def __init__(self, feats, window_size=None, prob=False):
		super(MAD_GAN, self).__init__()
		self.name = 'MAD_GAN'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_hidden = 16
		self.window_size = 5 # MAD_GAN w_size = 5
		self.n = self.n_feats * self.window_size
		self.generator = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.discriminator = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, 1), nn.Sigmoid(),
		)

	def forward(self, g):
		## Generate
		z = self.generator(g.view(1,-1))
		## Discriminator
		real_score = self.discriminator(g.view(1,-1))
		fake_score = self.discriminator(z.view(1,-1))
		return z.view(-1), real_score.view(-1), fake_score.view(-1)

# Proposed Model (VLDB 22)
class TranAD_Basic(nn.Module):
	def __init__(self, feats):
		super(TranAD_Basic, self).__init__()
		self.name = 'TranAD_Basic'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.window_size = 10
		self.n = self.n_feats * self.window_size
		self.pos_encoder = PositionalEncoding(feats, 0.1, self.window_size)
		encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()

	def forward(self, src, tgt):
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		x = self.transformer_decoder(tgt, memory)
		x = self.fcn(x)
		return x

# Proposed Model (FCN) + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD_Transformer(nn.Module):
	def __init__(self, feats):
		super(TranAD_Transformer, self).__init__()
		self.name = 'TranAD_Transformer'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_hidden = 8
		self.window_size = 10
		self.n = 2 * self.n_feats * self.window_size
		self.transformer_encoder = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.ReLU(True))
		self.transformer_decoder1 = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, 2 * feats), nn.ReLU(True))
		self.transformer_decoder2 = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, 2 * feats), nn.ReLU(True))
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src.permute(1, 0, 2).flatten(start_dim=1)
		tgt = self.transformer_encoder(src)
		return tgt

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.transformer_decoder1(self.encode(src, c, tgt))
		x1 = x1.reshape(-1, 1, 2*self.n_feats).permute(1, 0, 2)
		x1 = self.fcn(x1)
		# Phase 2 - With anomaly scores
		c = (x1 - src) ** 2
		x2 = self.transformer_decoder2(self.encode(src, c, tgt))
		x2 = x2.reshape(-1, 1, 2*self.n_feats).permute(1, 0, 2)
		x2 = self.fcn(x2)
		return x1, x2

# Proposed Model + Self Conditioning + MAML (VLDB 22)
class TranAD_Adversarial(nn.Module):
	def __init__(self, feats):
		super(TranAD_Adversarial, self).__init__()
		self.name = 'TranAD_Adversarial'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.window_size = 10
		self.n = self.n_feats * self.window_size
		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.window_size)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

	def encode_decode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		x = self.transformer_decoder(tgt, memory)
		x = self.fcn(x)
		return x

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x = self.encode_decode(src, c, tgt)
		# Phase 2 - With anomaly scores
		c = (x - src) ** 2
		x = self.encode_decode(src, c, tgt)
		return x

# Proposed Model + Adversarial + MAML (VLDB 22)
class TranAD_SelfConditioning(nn.Module):
	def __init__(self, feats):
		super(TranAD_SelfConditioning, self).__init__()
		self.name = 'TranAD_SelfConditioning'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.window_size = 10
		self.n = self.n_feats * self.window_size
		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.window_size)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
		# Phase 2 - With anomaly scores
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
		return x1, x2

# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(nn.Module):
	def __init__(self, feats, window_size, prob=False):
		super(TranAD, self).__init__()
		self.name = 'TranAD'
		self.lr = lr
		self.batch = 24 # 1 if window_size > 1280 else int(1280 / window_size) # 128
		self.n_feats = feats
		self.window_size = window_size
		self.n = self.n_feats * self.window_size
		self.prob = prob
		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.window_size)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)  #, enable_nested_tensor=False)
		decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		if self.prob:
			self.fcn = nn.Sequential(nn.Linear(2 * feats, 2 * feats), nn.Sigmoid())  # double the outputs
		else:
			self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1_out = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
		if self.prob:
			x1_mu, x1_logsigma = torch.split(x1_out, split_size_or_sections=self.n_feats, dim=2)
			x1 = x1_mu + torch.randn(size=x1_logsigma.size()) * torch.exp(x1_logsigma)
		else:
			x1 = x1_out
		# Phase 2 - With anomaly scores
		c = (x1 - src) ** 2
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
		return x1_out, x2

class iTransformer(nn.Module):
	def __init__(self, feats, window_size, step_size=None, prob=False, weighted_window=False, forecasting=False):
		super(iTransformer, self).__init__()
		self.name = 'iTransformer'
		self.weighted = weighted_window
		self.lr = lr
		self.batch = 4  #  if window_size > 1000 else int(1280 / window_size) # 128 for window size 10
		self.n_feats = feats
		self.window_size = window_size
		if step_size is not None:
			self.test_step_size = step_size
		else:
			self.test_step_size = window_size
		self.n = self.n_feats * self.window_size
		self.seq_len = self.window_size
		self.label_len = self.window_size
		self.forecasting = forecasting 		# whether model output should be reconstruction or forecasting of input
		if self.forecasting:
			self.pred_len = 1  				# for forecasting-based AD, only want to predict 1 step ahead
		else:
			self.pred_len = self.window_size
		self.output_attention = False
		self.use_norm = True
		self.d_model = 2  # int(self.window_size / 2) # * feats  # 512
		self.embed = 'TimeF'
		self.freq = 's'
		self.dropout = 0.1
		self.n_heads = feats  # was done like this for other algos
		self.e_layers = 2
		self.d_ff = 128 # 128 # 16
		self.factor = 1  # attention factor
		self.activation = 'gelu'
		self.prob = prob 		# whether model gives back probabilistic output instead of single value
        # Embedding
		self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, self.embed, self.freq, self.dropout)
        # Encoder-only architecture
		self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), 
									  self.d_model, self.n_heads, d_keys=self.d_model, d_values=self.d_model),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
		if self.prob:
			self.projector = nn.Linear(self.d_model, 2*self.pred_len, bias=True)  # double the outputs
		else:
			self.projector = nn.Linear(self.d_model, self.pred_len, bias=True)

	def forecast(self, x_enc, x_mark_enc=None):
		if self.use_norm:
			# Normalization from Non-stationary Transformer
			means = x_enc.mean(1, keepdim=True).detach()
			x_enc = x_enc - means
			stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
			x_enc /= stdev

		_, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
		enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
		enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N  
		dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

		if self.use_norm:
			# De-Normalization from Non-stationary Transformer
			if self.prob:
				dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, 2*self.pred_len, 1))
				dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, 2*self.pred_len, 1))
			else:
				dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
				dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

		return dec_out
	
	def forward(self, src, src_mark_enc=None):
		# decoder input
		# dec_inp = torch.zeros_like(src[:, -self.pred_len:, :])
		# dec_inp = torch.cat([src[:, :self.label_len, :], dec_inp], dim=1)
		
		dec_out = self.forecast(src, src_mark_enc)  # [B, 2L, N]
		if self.prob:
			dec_mu, dec_logsigma = torch.split(dec_out, split_size_or_sections=self.pred_len, dim=1)
			# dec_out = dec_mu + torch.randn(size=dec_logsigma.size())*torch.exp(dec_logsigma)
			dec_mu = dec_mu[:, -1:, :].permute(1, 0, 2)  # [1, B, N], for AD only give back last element of window/sequence
			dec_logsigma = dec_logsigma[:, -1:, :].permute(1, 0, 2)  # [1, B, N], for AD only give back last element of window/sequence
			return dec_mu, dec_logsigma
		else:
			# dec_out = dec_out[:, :, :]  # [B, 1, N], for AD only give back last element of window/sequence
			# out = dec_out.permute(1, 0, 2)  # [1, B, N], permute to have same output structure as other models
			out = dec_out
			return out

# iTransformer with encoder + decoder structure
class iTransformer_dec(nn.Module):
	def __init__(self, feats, window_size, prob=False):
		super(iTransformer_dec, self).__init__()
		self.name = 'iTransformer_dec'
		self.lr = lr
		self.batch = 2 # if window_size > 1280 else int(1280 / window_size) # 128 for window size 10
		self.n_feats = feats
		self.window_size = window_size
		self.n = self.n_feats * self.window_size
		self.seq_len = self.window_size
		self.label_len = self.window_size
		self.pred_len = self.window_size
		self.output_attention = False
		self.use_norm = True
		self.d_model = 2 # 512
		self.embed = 'TimeF'
		self.freq = 's'
		self.dropout = 0.1
		self.n_heads = 5  # was done like this for other algos
		self.e_layers = 1  # encoder layers
		self.d_layers = 1  # decoder layers
		self.d_ff = 128 # 128 # 16
		self.latent = self.d_model  # dimension of latent space
		self.factor = 1  # attention factor
		self.activation = 'gelu'
		self.prob = prob 		# whether model gives back probabilistic output instead of single value
        # Embedding
		self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, self.embed, self.freq, self.dropout)
        # Encoder+decoder architecture
		self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), 
									  self.d_model, self.n_heads, d_keys=self.d_model, d_values=self.d_model),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
		
		self.projector = nn.Linear(self.d_model, self.latent, bias=True)
		self.dec_embedding = DataEmbedding_inverted(self.seq_len, self.latent, self.embed, self.freq, self.dropout)
	
		self.decoder = Decoder(
		[
			DecoderLayer(
				AttentionLayer(  # self-attention
					FullAttention(True, self.factor, attention_dropout=self.dropout,
									output_attention=False),
					self.latent, self.n_heads, d_keys=self.latent, d_values=self.latent),
				None,  # cross-attention
				# AttentionLayer(  # cross-attention
				# 	FullAttention(False, self.factor, attention_dropout=self.dropout,
				# 					output_attention=False),
				# 	self.latent, self.n_heads, d_keys=self.latent, d_values=self.latent),
				self.latent,
				self.d_ff,
				dropout=self.dropout,
				activation=self.activation,
			)
			for l in range(self.d_layers)
		],
		norm_layer=torch.nn.LayerNorm(self.latent),
		projection = nn.Linear(self.latent, self.pred_len, bias=True)
		)
		# else:
		# 	decoder_layers = TransformerDecoderLayer(d_model=self.d_model, nhead=self.n_feats, dim_feedforward=self.d_ff, dropout=0.1)
		# 	self.decoder= TransformerDecoder(decoder_layers, 1)
		

	def forecast(self, x_enc, x_mark_enc=None):
		if self.use_norm:
			# Normalization from Non-stationary Transformer
			means = x_enc.mean(1, keepdim=True).detach()
			x_enc = x_enc - means
			stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
			x_enc /= stdev

		_, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
		enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
		# B L N -> B N E/latent 
		dec_out = self.dec_embedding(x_enc, x_mark_enc)
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
		enc_out, attns = self.encoder(enc_out, attn_mask=None)
		# B N E -> B N E/latent
		enc_out = self.projector(enc_out)
		 
		# B N E/latent, B N E/latent -> B N E -> B N S
		dec_out = self.decoder(enc_out, None)

        # B N S -> B S N
		dec_out = dec_out.permute(0, 2, 1)[:, :, :N] # filter the covariates

		if self.use_norm:
			# De-Normalization from Non-stationary Transformer
			dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
			dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

		return dec_out
	
	def forward(self, src, src_mark_enc=None):

		dec_out = self.forecast(src, src_mark_enc)  # [B, L, N]
		return dec_out