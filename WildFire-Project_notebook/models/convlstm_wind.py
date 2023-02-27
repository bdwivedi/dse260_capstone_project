import torch
import torch.nn as nn

class ConvLSTM_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTM_Cell, self).__init__()

        self.height, self.width = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)

    # we implement LSTM that process only one timestep
    def forward(self, x, hidden):  # x [batch, hidden_dim, width, height]
        h_cur, c_cur = hidden

        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dims, n_layers, kernel_size, device):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [], []
        self.device = device

        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            print('layer ', i, 'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(ConvLSTM_Cell(input_shape=self.input_shape,
                                           input_dim=cur_input_dim,
                                           hidden_dim=self.hidden_dims[i],
                                           kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False, HC=None):  # input_ [batch_size, 1, channels, width, height]
        batch_size = input_.data.size()[0]
        if (first_timestep):
            HC = self.initHidden(batch_size)  # init Hidden at each forward start
        H, C = HC
        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                H[j], C[j] = cell(input_, (H[j], C[j]))
            else:
                H[j], C[j] = cell(H[j - 1], (H[j], C[j]))

        return (H, C), H  # (hidden, output)

    def initHidden(self, batch_size):
        H, C = [], []
        for i in range(self.n_layers):
            H.append(
                torch.zeros(batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device))
            C.append(
                torch.zeros(batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device))
        return H, C

    def setHidden(self, hidden):
        H, C = hidden
        self.H, self.C = H, C


class dcgan_conv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3), stride=stride, padding=1),
            nn.GroupNorm(16, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if (stride == 2):
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3), stride=stride, padding=1,
                               output_padding=output_padding),
            nn.GroupNorm(16, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class encoder_E(nn.Module):
    def __init__(self, nc=3, nf=32):
        super(encoder_E, self).__init__()
      
        self.c1 = dcgan_conv(nc, nf, stride=2) 
        self.c2 = dcgan_conv(nf, nf, stride=1)  
        self.c3 = dcgan_conv(nf, 2 * nf, stride=2)  

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3


class decoder_D(nn.Module):
    def __init__(self, nc=1, nf=32):
        super(decoder_D, self).__init__()
        self.upc1 = dcgan_upconv(2 * nf, nf, stride=2) 
        self.upc2 = dcgan_upconv(nf, nf, stride=1)  
        self.upc3 = nn.ConvTranspose2d(in_channels=nf, out_channels=nc, kernel_size=(3, 3), stride=2, padding=1,
                                       output_padding=1) 

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        return d3


class encoder_specific(nn.Module):
    def __init__(self, nc=64, nf=64):
        super(encoder_specific, self).__init__()
        self.c1 = dcgan_conv(nc, nf, stride=1)  
        self.c2 = dcgan_conv(nf, nf, stride=1) 
    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        return h2


class decoder_specific(nn.Module):
    def __init__(self, nc=64, nf=64):
        super(decoder_specific, self).__init__()
        self.upc1 = dcgan_upconv(nf, nf, stride=1)  
        self.upc2 = dcgan_upconv(nf, nc, stride=1) 

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        return d2


class EncoderRNN(torch.nn.Module):
    """
    Modified from phydnet to build convlstm.
    """

    def __init__(self,  convcell, device):
        super(EncoderRNN, self).__init__()
        self.encoder_E = encoder_E()  
        #self.encoder_Ep = encoder_specific()  
        self.encoder_Er = encoder_specific()
        #self.decoder_Dp = decoder_specific() 
        self.decoder_Dr = decoder_specific()
        self.decoder_D = decoder_D() 

        self.encoder_E = self.encoder_E.to(device)
        #self.encoder_Ep = self.encoder_Ep.to(device)
        self.encoder_Er = self.encoder_Er.to(device)
        #self.decoder_Dp = self.decoder_Dp.to(device)
        self.decoder_Dr = self.decoder_Dr.to(device)
        self.decoder_D = self.decoder_D.to(device)
        self.convcell = convcell.to(device)
        self.relu=nn.ReLU()
        self.activation = torch.sigmoid

    def forward(self, input, first_timestep=False, decoding=False, hidden=None):
        input2 = self.encoder_E(input) 

        input_conv = self.encoder_Er(input2)

        hidden, output = self.convcell(input_conv, first_timestep, HC=hidden)

        decoded_Dr = self.decoder_Dr(output[-1])

        output_image = self.relu(input[:,:-2]-self.activation(self.decoder_D(decoded_Dr)))
        return hidden, output_image