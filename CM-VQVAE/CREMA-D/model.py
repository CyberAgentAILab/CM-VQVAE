import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)

        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        batch_size = inputs.shape[0]

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, quantized.view(batch_size,
                                                                                            -1)  # self._embedding.weight.clone().detach()


class VectorQuantizerEMA(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        batch_size = inputs.shape[0]

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, quantized.view(batch_size,
                                                                                            -1)  # self._embedding.weight.clone().detach()


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_residual_hiddens),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])
        self._bn = nn.BatchNorm2d(num_hiddens)

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(self._bn(x))


class EncoderImage(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(EncoderImage, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._bn_1 = nn.BatchNorm2d(num_hiddens // 2)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._bn_2 = nn.BatchNorm2d(num_hiddens)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(self._bn_1(x))

        x = self._conv_2(x)
        x = F.relu(self._bn_2(x))

        x = self._conv_3(x)
        return self._residual_stack(x)

    
class EncoderAudio(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(EncoderAudio, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._bn_1 = nn.BatchNorm2d(num_hiddens // 2)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._bn_2 = nn.BatchNorm2d(num_hiddens)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(self._bn_1(x))

        x = self._conv_2(x)
        x = F.relu(self._bn_2(x))

        x = self.maxpool(self._conv_3(x))
        return self._residual_stack(x)
    

class DecoderImage(nn.Module):
    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(DecoderImage, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)
        self._bn = nn.BatchNorm2d(num_hiddens // 2)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=out_channels,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(self._bn(x))

        return self._conv_trans_2(x)

    
class DecoderAudio(nn.Module):
    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(DecoderAudio, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)
        self._bn_1 = nn.BatchNorm2d(num_hiddens // 2)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=num_hiddens // 4,
                                                kernel_size=4,
                                                stride=2, padding=1)
        self._bn_2 = nn.BatchNorm2d(num_hiddens // 4)
        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens // 4,
                                                out_channels=out_channels,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(self._bn_1(x))
        x = self._conv_trans_2(x)
        x = F.relu(self._bn_2(x))
        
        return self._conv_trans_3(x)
    

# Code from: https://github.com/arunmallya/piggyback/blob/master/src/modnets/layers.py
DEFAULT_THRESHOLD = 5e-3


# Code from: https://discuss.pytorch.org/t/custom-autograd-function-must-it-be-static/14980/3
class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    #    def __init__(self, threshold=DEFAULT_THRESHOLD):
    #        super(Binarizer, self).__init__()
    #        self.threshold = threshold
    @staticmethod
    def forward(ctx, inputs, threshold):
        ctx.threshold = threshold
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1
        return outputs

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None


class Binarizer_Inverse(torch.autograd.Function):
    """Binarizes {1, 0} a real valued tensor."""

    @staticmethod
    def forward(ctx, inputs, threshold):
        ctx.threshold = threshold
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 1
        outputs[inputs.gt(threshold)] = 0
        return outputs

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None
    
    
class Ternarizer(torch.autograd.Function):
    """Ternarizes {-1, 0, 1} a real valued tensor."""

    #    def __init__(self, threshold=DEFAULT_THRESHOLD):
    #        super(Ternarizer, self).__init__()
    #        self.threshold = threshold

    @staticmethod
    def forward(ctx, inputs, threshold):
        outputs = inputs.clone()
        outputs.fill_(0)
        outputs[inputs < 0] = -1
        outputs[inputs > threshold] = 1
        return outputs

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None


class Randomizer(torch.autograd.Function):
    """Randomizes {0, 1} a real valued tensor."""

    #    def __init__(self, threshold=DEFAULT_THRESHOLD):
    #        super(Binarizer, self).__init__()
    #        self.threshold = threshold
    @staticmethod
    def forward(ctx, inputs, threshold):
        outputs = inputs.clone()
        dropout = torch.rand(inputs.shape)#.to(torch.device("cuda")))
        outputs[dropout.le(threshold)] = 0
        outputs[dropout.gt(threshold)] = 1
        return outputs

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None


class TaskSolver(nn.Module):
    def __init__(self, mode, in_channels, out_channels, num_hiddens, embedding_dim, num_residual_layers,
                 num_residual_hiddens, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer', threshold=None):
        super(TaskSolver, self).__init__()

        self.mode = mode
        if self.mode == 'multimodal':
            in_size = in_channels * 2
        else:
            in_size = in_channels
        
        self._conv1 = nn.Conv2d(in_channels=in_size,
                                out_channels=int(in_channels / 2),
                                kernel_size=3,
                                stride=2, padding=1)
        self._bn_1 = nn.BatchNorm2d(int(in_channels / 2))
        self._conv2 = nn.Conv2d(in_channels=int(in_channels / 2),
                                out_channels=int(in_channels / 2),
                                kernel_size=3,
                                stride=2, padding=1)
        self._conv_drop = nn.Dropout()

        self._residual_stack = ResidualStack(in_channels=int(num_hiddens / 4),
                                             num_hiddens=int(num_hiddens / 4),
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=int(num_residual_hiddens / 4))

        self._fc1 = nn.Linear(2048, 256)
        self._bn_2 = nn.BatchNorm1d(256)
        self._fc2 = nn.Linear(256, out_channels)

        if threshold is None:
            self.threshold = DEFAULT_THRESHOLD
        else:
            self.threshold = threshold
        self.threshold_fn = threshold_fn

        # Initialize learnable mask (0s will represent private codes, and 1s will represent shared codes)
        self.private_shared_mask = torch.zeros(in_channels * 2, 1, 1)
        if mask_init == '1s':
            self.private_shared_mask.fill_(mask_scale)
        elif mask_init == 'uniform':
            self.private_shared_mask.uniform_(-1 * mask_scale, mask_scale)
        # private_shared_mask is now a trainable parameter.
        self.private_shared_mask = nn.parameter.Parameter(self.private_shared_mask)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        num_channels = int(inputs.shape[1]/2)
        
        # Get binarized/ternarized mask
        if self.threshold_fn == 'binarizer':
            mask_thresholded = Binarizer.apply(self.private_shared_mask, self.threshold)
        elif self.threshold_fn == 'ternarizer':
            mask_thresholded = Ternarizer.apply(self.private_shared_mask, self.threshold)
        elif self.threshold_fn == 'attention':
            mask_thresholded = self.private_shared_mask
        elif self.threshold_fn == 'randomizer':
            mask_thresholded = Randomizer.apply(self.private_shared_mask, 0.5)
        else:
            print("ERROR - Unknown mask thresholding method: {}".format(self.threshold_fn))
            exit()
        # print("Inputs: {}, Mask: {}".format(inputs.size(), mask_thresholded.size()))
        # mask_thresholded = mask_thresholded.unsqueeze(1).repeat(1, inputs.size()[1])
        # inputs = inputs.unsqueeze(0).repeat(batch_size, 1, 1)
        # mask_thresholded = mask_thresholded.unsqueeze(0).repeat(batch_size, 1, 1)
        # print("Inputs: {}, Mask: {}".format(inputs.size(), mask_thresholded.size()));exit()
        # Mask weights with above mask
        x = torch.mul(inputs, mask_thresholded)
        # Make each code a channel for the convolution: BDC -> BCD (H = 1)
        # inputs = inputs.permute(0, 2, 1).contiguous()

        if self.mode == 'image-only':
            x = x[:,:num_channels,:,:]
        elif self.mode == 'audio-only':
            x = x[:,num_channels:,:,:]
        
        x = F.relu(self._bn_1(self._conv1(x)))
        x = self._conv_drop(self._conv2(x))
        x = self._residual_stack(x)
        x = x.view(batch_size, -1)
        x = F.relu(self._bn_2(self._fc1(x)))
        x = F.dropout(x, training=self.training)
        x = self._fc2(x)
        y = x  # F.log_softmax(x, dim=-1)

        _, _, _, _, _, shared_loss = self.get_private_shared_ratio()  # Regularization term so mask is not fully ones

        return shared_loss, y

    # def _apply(self, fn):
    #    for module in self.children():
    #        module._apply(fn)
    #
    #    for param in self._parameters.values():
    #        if param is not None:
    #            # Variables stored in modules are graph leaves, and we don't
    #            # want to create copy nodes, so we have to unpack the datasets.
    #            param.datasets = fn(param.datasets)
    #            if param._grad is not None:
    #                param._grad.datasets = fn(param._grad.datasets)
    #
    #    for key, buf in self._buffers.items():
    #        if buf is not None:
    #            self._buffers[key] = fn(buf)
    #
    #    self.weight.datasets = fn(self.weight.datasets)
    #    self.bias.datasets = fn(self.bias.datasets)

    def get_private_shared_ratio(self):
        # Get binarized/ternarized mask
        if self.threshold_fn == 'binarizer':
            mask_thresholded = Binarizer.apply(self.private_shared_mask, self.threshold)
        elif self.threshold_fn == 'ternarizer':
            mask_thresholded = Ternarizer.apply(self.private_shared_mask, self.threshold)
        elif self.threshold_fn == 'attention':
            mask_thresholded = Binarizer.apply(self.private_shared_mask, self.threshold)
        elif self.threshold_fn == 'randomizer':
            mask_thresholded = Randomizer.apply(self.private_shared_mask, 0.5)
        else:
            print("ERROR - Unknown mask thresholding method: {}".format(self.threshold_fn))
            exit()

        total_codes = len(mask_thresholded)
        mod1_mask = mask_thresholded[:int(total_codes / 2)]  # Variable that stores the values to mask modality 1
        mod2_mask = mask_thresholded[int(total_codes / 2):]  # Variable that stores the values to mask modality 2

        private1_codes = len(mod1_mask) - torch.count_nonzero(mod1_mask)
        private2_codes = len(mod2_mask) - torch.count_nonzero(mod2_mask)
        #shared_codes = total_codes - private1_codes - private2_codes
        shared1_codes = len(mod1_mask) - private1_codes
        shared2_codes = len(mod2_mask) - private2_codes

        private1_ratio = private1_codes / total_codes * 100
        private2_ratio = private2_codes / total_codes * 100
        #shared_ratio = shared_codes / total_codes * 100
        shared1_ratio = shared1_codes / total_codes * 100
        shared2_ratio = shared2_codes / total_codes * 100
        
        if shared1_codes < shared2_codes:
            complementarity = shared1_codes / shared2_codes
        else:
            complementarity = shared2_codes / shared1_codes

        shared_loss = torch.sum(self.private_shared_mask)
        
        return private1_ratio, private2_ratio, shared1_ratio, shared2_ratio, complementarity, shared_loss

    def get_mask(self):
        return self.private_shared_mask
    
    def reset_mask(self):
        self.private_shared_mask.data.fill_(1e-2)
    

class Model(nn.Module):
    def __init__(self, dataset, mode, num_hiddens, num_classes, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay, mask_init, mask_scale, threshold_fn, threshold):
        super(Model, self).__init__()

        self._encoder_image = EncoderImage(3, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._decoder_image = DecoderImage(embedding_dim, 3, num_hiddens, num_residual_layers, num_residual_hiddens)
        if dataset == 'EmoVoxCeleb':
            self._encoder_audio = EncoderImage(1, num_hiddens, num_residual_layers, num_residual_hiddens)
            self._decoder_audio = DecoderImage(embedding_dim, 1, num_hiddens, num_residual_layers, num_residual_hiddens)
        else:
            self._encoder_audio = EncoderAudio(1, num_hiddens, num_residual_layers, num_residual_hiddens)
            self._decoder_audio = DecoderAudio(embedding_dim, 1, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv_image = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)
        self._pre_vq_conv_audio = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)
        if decay > 0.0:
            self._vq_vae_image = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
            self._vq_vae_audio = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self._vq_vae_image = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            self._vq_vae_audio = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self._classifier = TaskSolver(mode, embedding_dim, num_classes, num_hiddens, num_embeddings,
                                      num_residual_layers, num_residual_hiddens, mask_init, mask_scale, threshold_fn,
                                      threshold)

    def forward(self, x_image, x_audio):
        z_image = self._encoder_image(x_image)
        z_audio = self._encoder_audio(x_audio)
        z_image = self._pre_vq_conv_image(z_image)
        z_audio = self._pre_vq_conv_audio(z_audio)
        loss_image, quantized_image, perplexity_image, codes_image = self._vq_vae_image(z_image)
        loss_audio, quantized_audio, perplexity_audio, codes_audio = self._vq_vae_audio(z_audio)
        # print("MNIST quant: {}, SVHN quant: {}".format(quantized_mnist.size(), quantized_svhn.size()))
        # print("MNIST codes: {}, SVHN codes: {}".format(codes_mnist.size(), codes_svhn.size()))
        x_recon_image = self._decoder_image(quantized_image)
        x_recon_audio = self._decoder_audio(quantized_audio)
        loss_shared, y = self._classifier(torch.cat((quantized_image, quantized_audio), dim=1))

        return loss_image, loss_audio, x_recon_image, x_recon_audio, loss_shared, y, perplexity_image, perplexity_audio

    
class Model_VQVAE(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay):
        super(Model_VQVAE, self).__init__()

        self._encoder_image = EncoderImage(3, num_hiddens,
                                      num_residual_layers,
                                      num_residual_hiddens)
        self._encoder_audio = EncoderAudio(1, num_hiddens,
                                     num_residual_layers,
                                     num_residual_hiddens)
        self._pre_vq_conv_image = nn.Conv2d(in_channels=num_hiddens,
                                            out_channels=embedding_dim,
                                            kernel_size=1,
                                            stride=1)
        self._pre_vq_conv_audio = nn.Conv2d(in_channels=num_hiddens,
                                           out_channels=embedding_dim,
                                           kernel_size=1,
                                           stride=1)
        if decay > 0.0:
            self._vq_vae_image = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
            self._vq_vae_audio = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self._vq_vae_image = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            self._vq_vae_audio = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self._decoder_image = DecoderImage(embedding_dim, 3,
                                      num_hiddens,
                                      num_residual_layers,
                                      num_residual_hiddens)

        self._decoder_audio = DecoderAudio(embedding_dim, 1,
                                     num_hiddens,
                                     num_residual_layers,
                                     num_residual_hiddens)

    def forward(self, x_image, x_audio):
        z_image = self._encoder_image(x_image)
        z_audio = self._encoder_audio(x_audio)
        z_image = self._pre_vq_conv_image(z_image)
        z_audio = self._pre_vq_conv_audio(z_audio)
        loss_image, quantized_image, perplexity_image, codes_image = self._vq_vae_image(z_image)
        loss_audio, quantized_audio, perplexity_audio, codes_audio = self._vq_vae_audio(z_audio)
        # print("MNIST quant: {}, SVHN quant: {}".format(quantized_mnist.size(), quantized_svhn.size()))
        # print("MNIST codes: {}, SVHN codes: {}".format(codes_mnist.size(), codes_svhn.size()))
        x_recon_image = self._decoder_image(quantized_image)
        x_recon_audio = self._decoder_audio(quantized_audio)

        return loss_image, loss_audio, x_recon_image, x_recon_audio, quantized_image, quantized_audio, perplexity_image, perplexity_audio

    
class Model_Solver(nn.Module):
    def __init__(self, num_hiddens, num_classes, num_residual_layers, num_residual_hiddens, num_embeddings,
                 embedding_dim, mask_init, mask_scale, threshold_fn, threshold):
        super(Model_Solver, self).__init__()

        self._classifier = TaskSolver(embedding_dim, num_classes, num_hiddens, num_embeddings,
                                      num_residual_layers, num_residual_hiddens, mask_init, mask_scale, threshold_fn,
                                      threshold)

    def forward(self, quantized_image, quantized_audio):

        loss_shared, y = self._classifier(torch.cat((quantized_image, quantized_audio), dim=1))

        return loss_shared, y

    
class Model_Feat(nn.Module):
    def __init__(self, dataset, num_hiddens, num_classes, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay, mask_init, mask_scale, threshold_fn, threshold):
        super(Model_Feat, self).__init__()
        
        if decay > 0.0:
            self._vq_vae_image = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
            self._vq_vae_audio = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self._vq_vae_image = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            self._vq_vae_audio = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self._classifier = TaskSolver_Feat(embedding_dim, num_classes, num_hiddens, num_embeddings,
                                      num_residual_layers, num_residual_hiddens, mask_init, mask_scale, threshold_fn,
                                      threshold)

    def forward(self, x_image, x_audio):
        batch_size = x_image.shape[0]
        
        z_image = x_image.view(batch_size, 64, 32, 32)  # Adapt dimensions to VQVAE bottleneck
        z_audio = x_audio.view(batch_size, 64, 32, 32)  # Adapt dimensions to VQVAE bottleneck

        loss_image, quantized_image, perplexity_image, codes_image = self._vq_vae_image(z_image)
        loss_audio, quantized_audio, perplexity_audio, codes_audio = self._vq_vae_audio(z_audio)
        
        x_recon_image = quantized_image.view(batch_size, 1, 16, 4096)  # Get original dimensions back
        x_recon_audio = quantized_audio.view(batch_size, 1, 64, 1024)  # Get original dimensions back

        loss_shared, y = self._classifier(torch.cat((quantized_image, quantized_audio), dim=1))

        return loss_image, loss_audio, x_recon_image, x_recon_audio, loss_shared, y, perplexity_image, perplexity_audio

    
class TaskSolver_Feat(nn.Module):
    def __init__(self, in_channels, out_channels, num_hiddens, embedding_dim, num_residual_layers,
                 num_residual_hiddens, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer', threshold=None):
        super(TaskSolver_Feat, self).__init__()

        # Input size: 65536 = 16x4096 & 64x1024
        
        self._fc_image_1 = nn.Linear(4096, 1024)
        self._bn_i_1 = nn.BatchNorm1d(1024)
        self._fc_image_2 = nn.Linear(1024, 512)
        self._bn_i_2 = nn.BatchNorm1d(512)
        self._fc_audio_1 = nn.Linear(1024, 512)
        self._bn_a_1 = nn.BatchNorm1d(512)

        input_size = 1024
        hidden_size = 256
        self._lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self._bn_f = nn.BatchNorm1d(256)
        
        self._fc_last = nn.Linear(256, 6)

        if threshold is None:
            self.threshold = DEFAULT_THRESHOLD
        else:
            self.threshold = threshold
        self.threshold_fn = threshold_fn

        # Initialize learnable mask (0s will represent private codes, and 1s will represent shared codes)
        self.private_shared_mask = torch.zeros(in_channels * 2, 1, 1)
        if mask_init == '1s':
            self.private_shared_mask.fill_(mask_scale)
        elif mask_init == 'uniform':
            self.private_shared_mask.uniform_(-1 * mask_scale, mask_scale)
        # private_shared_mask is now a trainable parameter.
        self.private_shared_mask = nn.parameter.Parameter(self.private_shared_mask)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        num_channels = int(inputs.shape[1]/2)
        
        # Get binarized/ternarized mask
        if self.threshold_fn == 'binarizer':
            mask_thresholded = Binarizer.apply(self.private_shared_mask, self.threshold)
        elif self.threshold_fn == 'ternarizer':
            mask_thresholded = Ternarizer.apply(self.private_shared_mask, self.threshold)
        # print("Inputs: {}, Mask: {}".format(inputs.size(), mask_thresholded.size()))
        # mask_thresholded = mask_thresholded.unsqueeze(1).repeat(1, inputs.size()[1])
        # inputs = inputs.unsqueeze(0).repeat(batch_size, 1, 1)
        # mask_thresholded = mask_thresholded.unsqueeze(0).repeat(batch_size, 1, 1)
        # print("Inputs: {}, Mask: {}".format(inputs.size(), mask_thresholded.size()));exit()
        # Mask weights with above mask
        x = torch.mul(inputs, mask_thresholded)
        # Make each code a channel for the convolution: BDC -> BCD (H = 1)
        # inputs = inputs.permute(0, 2, 1).contiguous()

        x_image = x[:,:num_channels,:,:].view(batch_size, 16, 4096)
        x_audio = x[:,num_channels:,:,:].view(batch_size, 64, 1024)
        x_audio = self.subsample_features(x_audio)
        
        x_image = self._bn_i_1(self._fc_image_1(x_image).permute(0, 2, 1)).permute(0, 2, 1)
        x_image = F.dropout(F.relu(x_image), 0.5, self.training)
        x_image = self._bn_i_2(self._fc_image_2(x_image).permute(0, 2, 1)).permute(0, 2, 1)
        x_image = F.dropout(F.relu(x_image), 0.5, self.training)
        
        x_audio = self._bn_a_1(self._fc_audio_1(x_audio).permute(0, 2, 1)).permute(0, 2, 1)
        x_audio = F.dropout(F.relu(x_audio), 0.5, self.training)
        
        x_fusion = torch.cat((x_image, x_audio), dim=-1)
        output, _ = self._lstm(x_fusion)
        x_fusion = self._bn_f(output[:,-1,:])
        x_fusion = F.dropout(F.relu(x_fusion), 0.5, self.training)
        
        y = self._fc_last(x_fusion)

        _, _, _, _, _, shared_loss = self.get_private_shared_ratio()  # Regularization term so mask is not fully ones

        return shared_loss, y

    def subsample_features(self, features):
        # Subsample 64 time frames into 16
        subsample = 16
        step = (features.shape[1] - 1) / (subsample - 1)
        index = [round(step * i) for i in range(subsample)]

        return(features[:,index,:])
    # def _apply(self, fn):
    #    for module in self.children():
    #        module._apply(fn)
    #
    #    for param in self._parameters.values():
    #        if param is not None:
    #            # Variables stored in modules are graph leaves, and we don't
    #            # want to create copy nodes, so we have to unpack the datasets.
    #            param.datasets = fn(param.datasets)
    #            if param._grad is not None:
    #                param._grad.datasets = fn(param._grad.datasets)
    #
    #    for key, buf in self._buffers.items():
    #        if buf is not None:
    #            self._buffers[key] = fn(buf)
    #
    #    self.weight.datasets = fn(self.weight.datasets)
    #    self.bias.datasets = fn(self.bias.datasets)

    def get_private_shared_ratio(self):
        # Get binarized/ternarized mask
        if self.threshold_fn == 'binarizer':
            mask_thresholded = Binarizer.apply(self.private_shared_mask, self.threshold)
        elif self.threshold_fn == 'ternarizer':
            mask_thresholded = Ternarizer.apply(self.private_shared_mask, self.threshold)

        total_codes = len(mask_thresholded)
        mod1_mask = mask_thresholded[:int(total_codes / 2)]  # Variable that stores the values to mask modality 1
        mod2_mask = mask_thresholded[int(total_codes / 2):]  # Variable that stores the values to mask modality 2

        private1_codes = len(mod1_mask) - torch.count_nonzero(mod1_mask)
        private2_codes = len(mod2_mask) - torch.count_nonzero(mod2_mask)
        #shared_codes = total_codes - private1_codes - private2_codes
        shared1_codes = len(mod1_mask) - private1_codes
        shared2_codes = len(mod2_mask) - private2_codes

        private1_ratio = private1_codes / total_codes * 100
        private2_ratio = private2_codes / total_codes * 100
        #shared_ratio = shared_codes / total_codes * 100
        shared1_ratio = shared1_codes / total_codes * 100
        shared2_ratio = shared2_codes / total_codes * 100
        
        if shared1_codes < shared2_codes:
            complementarity = shared1_codes / shared2_codes
        else:
            complementarity = shared2_codes / shared1_codes

        shared_loss = torch.sum(self.private_shared_mask)
            
        return private1_ratio, private2_ratio, shared1_ratio, shared2_ratio, complementarity, shared_loss

    
class Model_Baseline(nn.Module):
    def __init__(self, dataset, mode, base_type, num_hiddens, num_classes, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay, mask_init, mask_scale, threshold_fn, threshold):
        super(Model_Baseline, self).__init__()

        self.base_type = base_type
        
        self._encoder_image = EncoderImage(3, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._decoder_image = DecoderImage(embedding_dim, 3, num_hiddens, num_residual_layers, num_residual_hiddens)
        if dataset == 'RML' or dataset == 'CREMA-D':
            self._encoder_audio = EncoderAudio(1, num_hiddens, num_residual_layers, num_residual_hiddens)
            self._decoder_audio = DecoderAudio(embedding_dim, 1, num_hiddens, num_residual_layers, num_residual_hiddens)
        else:
            print("ERROR - Unknown dataset: {}".format(dataset))
            exit()
        self._pre_vq_conv_image = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)
        self._pre_vq_conv_audio = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        if decay > 0.0:
            self._vq_vae_image = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
            self._vq_vae_audio = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self._vq_vae_image = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            self._vq_vae_audio = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        if self.base_type == 3:
            self._classifier = TaskSolver(mode, embedding_dim, num_classes, num_hiddens, num_embeddings,
                                      num_residual_layers, num_residual_hiddens, mask_init, mask_scale, threshold_fn,
                                      threshold)
        else:
            self._classifier = TaskSolver_Baseline(mode, embedding_dim, num_classes, num_hiddens, num_embeddings,
                                      num_residual_layers, num_residual_hiddens)

    def forward(self, x_image, x_audio):
        z_image = self._encoder_image(x_image)
        z_audio = self._encoder_audio(x_audio)
        z_image = self._pre_vq_conv_image(z_image)
        z_audio = self._pre_vq_conv_audio(z_audio)
        if self.base_type == 2:
            loss_image, z_image, perplexity_image, codes_image = self._vq_vae_image(z_image)
            loss_audio, z_audio, perplexity_audio, codes_audio = self._vq_vae_audio(z_audio)
            x_recon_image = self._decoder_image(z_image)
            x_recon_audio = self._decoder_audio(z_audio)
        else:
            loss_image = loss_audio = 0.0
            perplexity_image = perplexity_audio = torch.zeros(1)
            x_recon_image = x_image
            x_recon_audio = x_audio
        loss_shared, y = self._classifier(torch.cat((z_image, z_audio), dim=1))

        return loss_image, loss_audio, x_recon_image, x_recon_audio, loss_shared, y, perplexity_image, perplexity_audio


class TaskSolver_Baseline(nn.Module):
    def __init__(self, mode, in_channels, out_channels, num_hiddens, embedding_dim, num_residual_layers, num_residual_hiddens):
        super(TaskSolver_Baseline, self).__init__()

        self.mode = mode
        if self.mode == 'multimodal':
            in_size = in_channels * 2
        else:
            in_size = in_channels
        
        self._conv1 = nn.Conv2d(in_channels=in_size,
                                out_channels=int(in_channels / 2),
                                kernel_size=3,
                                stride=2, padding=1)
        self._bn_1 = nn.BatchNorm2d(int(in_channels / 2))
        self._conv2 = nn.Conv2d(in_channels=int(in_channels / 2),
                                out_channels=int(in_channels / 2),
                                kernel_size=3,
                                stride=2, padding=1)
        self._conv_drop = nn.Dropout()

        self._residual_stack = ResidualStack(in_channels=int(num_hiddens / 4),
                                             num_hiddens=int(num_hiddens / 4),
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=int(num_residual_hiddens / 4))

        self._fc1 = nn.Linear(2048, 256)
        self._bn_2 = nn.BatchNorm1d(256)
        self._fc2 = nn.Linear(256, out_channels)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        num_channels = int(inputs.shape[1]/2)

        if self.mode == 'image-only':
            inputs = inputs[:,:num_channels,:,:]
        elif self.mode == 'audio-only':
            inputs = inputs[:,num_channels:,:,:]
            
        x = F.relu(self._bn_1(self._conv1(inputs)))
        x = self._conv_drop(self._conv2(x))
        x = self._residual_stack(x)
        x = x.view(batch_size, -1)
        x = F.relu(self._bn_2(self._fc1(x)))
        x = F.dropout(x, training=self.training)
        x = self._fc2(x)
        y = x  # F.log_softmax(x, dim=-1)

        _, _, _, _, _, shared_loss = self.get_private_shared_ratio()  # Regularization term so mask is not fully ones

        return shared_loss, y

    def get_private_shared_ratio(self):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    
class Model_Probe(nn.Module):
    def __init__(self, num_hiddens, num_classes, batch_size, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay, mask_init, mask_scale, threshold_fn, threshold):
        super(Model_Probe, self).__init__()

        self._encoder_image = EncoderImage(3, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._encoder_audio = EncoderImage(1, num_hiddens,
                                            num_residual_layers,
                                            num_residual_hiddens)
        self._pre_vq_conv_image = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self._pre_vq_conv_audio = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae_image = VectorQuantizerEMA(batch_size, num_embeddings, embedding_dim, commitment_cost, decay)
            self._vq_vae_audio = VectorQuantizerEMA(batch_size, num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self._vq_vae_image = VectorQuantizer(batch_size, num_embeddings, embedding_dim, commitment_cost)
            self._vq_vae_audio = VectorQuantizer(batch_size, num_embeddings, embedding_dim, commitment_cost)

        self._decoder_image = DecoderImage(embedding_dim, 3,
                                            num_hiddens,
                                            num_residual_layers,
                                            num_residual_hiddens)

        self._decoder_audio = DecoderImage(embedding_dim, 1,
                                           num_hiddens,
                                           num_residual_layers,
                                           num_residual_hiddens)

        self._masker = Masker(embedding_dim, num_classes)

    def forward(self, x_image, x_audio, invert=False, mix=False):
        z_image = self._encoder_image(x_image)
        z_audio = self._encoder_audio(x_audio)
        z_image = self._pre_vq_conv_image(z_image)
        z_audio = self._pre_vq_conv_audio(z_audio)
        loss_image, quantized_image, perplexity_image, codes_image = self._vq_vae_image(z_image)
        loss_audio, quantized_audio, perplexity_audio, codes_audio = self._vq_vae_audio(z_audio)
        if not mix:
            masked_image, masked_audio = self._masker(torch.cat((quantized_image, quantized_audio), dim=1), invert)
        else:
            masked_image, masked_audio = self._masker.mix(torch.cat((quantized_image, quantized_audio), dim=1))
        x_recon_image = self._decoder_image(masked_image)
        x_recon_audio = self._decoder_audio(masked_audio)

        return loss_image, loss_audio, masked_image, masked_audio, x_recon_image, x_recon_audio, perplexity_image, perplexity_audio

    
class Masker(nn.Module):
    def __init__(self, in_channels, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer', threshold=None):
        super(Masker, self).__init__()
        
        self.in_size = in_channels

        if threshold is None:
            self.threshold = DEFAULT_THRESHOLD
        else:
            self.threshold = threshold
        self.threshold_fn = threshold_fn

        # Initialize learnable mask (0s will represent private codes, and 1s will represent shared codes)
        self.private_shared_mask = torch.zeros(in_channels * 2, 1, 1)
        if mask_init == '1s':
            self.private_shared_mask.fill_(mask_scale)
        elif mask_init == 'uniform':
            self.private_shared_mask.uniform_(-1 * mask_scale, mask_scale)
        # private_shared_mask is now a trainable parameter.
        self.private_shared_mask = nn.parameter.Parameter(self.private_shared_mask)

    def forward(self, inputs, invert=False):

        # Get binarized/ternarized mask
        if self.threshold_fn == 'binarizer':
            if not invert:
                mask_thresholded = Binarizer.apply(self.private_shared_mask, self.threshold)
            else:
                mask_thresholded = Binarizer_Inverse.apply(self.private_shared_mask, self.threshold)
        elif self.threshold_fn == 'ternarizer':
            mask_thresholded = Ternarizer.apply(self.private_shared_mask, self.threshold)
        masked = torch.mul(inputs, mask_thresholded)

        return masked[:,:self.in_size,:,:], masked[:,self.in_size:,:,:]
    
    def mix(self, inputs):
        
        # Get binarized/ternarized mask
        if self.threshold_fn == 'binarizer':
            mask_shared = Binarizer.apply(self.private_shared_mask, self.threshold)
            mask_private = Binarizer_Inverse.apply(self.private_shared_mask, self.threshold)
        elif self.threshold_fn == 'ternarizer':
            mask_shared = Ternarizer.apply(self.private_shared_mask, self.threshold)
            mask_private = Ternarizer.apply(self.private_shared_mask, self.threshold)
        masked_shared = torch.mul(inputs, mask_shared)
        masked_private = torch.mul(inputs, mask_private)
        # Exchange the private and shared features with the sample on its left
        masked = masked_shared + torch.roll(masked_private, 1, 0)

        return masked[:,:self.in_size,:,:], masked[:,self.in_size:,:,:]

    def get_private_shared_ratio(self):
        # Get binarized/ternarized mask
        if self.threshold_fn == 'binarizer':
            mask_thresholded = Binarizer.apply(self.private_shared_mask, self.threshold)
        elif self.threshold_fn == 'ternarizer':
            mask_thresholded = Ternarizer.apply(self.private_shared_mask, self.threshold)

        total_codes = len(mask_thresholded)
        mod1_mask = mask_thresholded[:int(total_codes/2)] # Variable that stores the values to mask modality 1
        mod2_mask = mask_thresholded[int(total_codes/2):] # Variable that stores the values to mask modality 2

        private1_codes = len(mod1_mask) - torch.count_nonzero(mod1_mask)
        private2_codes = len(mod2_mask) - torch.count_nonzero(mod2_mask)
        #shared_codes = total_codes - private1_codes - private2_codes
        shared1_codes = len(mod1_mask) - private1_codes
        shared2_codes = len(mod2_mask) - private2_codes

        private1_ratio = private1_codes / total_codes * 100
        private2_ratio = private2_codes / total_codes * 100
        #shared_ratio = shared_codes / total_codes * 100
        shared1_ratio = shared1_codes / total_codes * 100
        shared2_ratio = shared2_codes / total_codes * 100

        if shared1_codes < shared2_codes:
            complementarity = shared1_codes / shared2_codes
        else:
            complementarity = shared2_codes / shared1_codes

        shared_loss = torch.sum(self.private_shared_mask)
            
        return private1_ratio, private2_ratio, shared1_ratio, shared2_ratio, complementarity, shared_loss

    def set_mask(self, mask):
        self.private_shared_mask = mask


from torchvision.models import resnet18
class Dummy_Classifier():
    def get_private_shared_ratio(self):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
class Model_Comparison(nn.Module):
    
    def __init__(self, mode, num_classes):
        super(Model_Comparison, self).__init__()

        self.mode = mode
        self._classifier = Dummy_Classifier()
        
        self._resnet18_1 = resnet18()
        self._resnet18_2 = resnet18()
        self._resnet18_2.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        num_ftrs = 512#print(self._resnet18_1.fc.in_features)  # 
        
        self._resnet18_1.fc = nn.Identity()
        self._resnet18_2.fc = nn.Identity()
        
        if self.mode == 'multimodal':
            self._fc1 = nn.Linear(num_ftrs*2, 256)
        else:
            self._fc1 = nn.Linear(num_ftrs, 256)
        self._bn = nn.BatchNorm1d(256)
        self._fc2 = nn.Linear(256, num_classes)

    def forward(self, x_image, x_audio):
        z_image = self._resnet18_1(x_image)
        z_audio = self._resnet18_2(x_audio)
        
        if self.mode == 'image-only':
            y = self._fc1(z_image)
        elif self.mode == 'audio-only':
            y = self._fc1(z_audio)
        else:
            y = self._fc1(torch.cat((z_image, z_audio), dim=1))
        y = F.relu(self._bn(y))
        y = F.dropout(y, training=self.training)
        y = self._fc2(y)
        
        loss_image = loss_audio = loss_shared = 0.0
        perplexity_image = perplexity_audio = torch.zeros(1)
        x_recon_image = x_image
        x_recon_audio = x_audio

        return loss_image, loss_audio, x_recon_image, x_recon_audio, loss_shared, y, perplexity_image, perplexity_audio
