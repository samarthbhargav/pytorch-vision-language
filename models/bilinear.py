import torch
from torch import nn


class CompactBilinearPooling(nn.Module):
    """
    Compute compact bilinear pooling over two bottom inputs.

    Args:

        output_dim: output dimension for compact bilinear pooling.

        sum_pool: (Optional) If True, sum the output along height and width
                  dimensions and return output shape [batch_size, output_dim].
                  Otherwise return [batch_size, height, width, output_dim].
                  Default: True.

        rand_h_1: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_1`
                  if is None.

        rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_1`. Automatically generated from `seed_s_1` if is
                  None.

        rand_h_2: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_2`
                  if is None.

        rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_2`. Automatically generated from `seed_s_2` if is
                  None.
    """

    def __init__(self, input_dim1, input_dim2, output_dim,
                 sum_pool=True, cuda=True,
                 rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None):
        super(CompactBilinearPooling, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sum_pool = sum_pool
        self.device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')

        if rand_h_1 is None:
            torch.random.manual_seed(1)
            rand_h_1 = torch.randint(output_dim, size=(self.input_dim1,))
        if rand_s_1 is None:
            torch.random.manual_seed(3)
            rand_s_1 = 2 * torch.randint(2, size=(self.input_dim1,)) - 1

        self.sparse_sketch_matrix1 = self.generate_sketch_matrix(
            rand_h_1, rand_s_1, self.output_dim)

        if rand_h_2 is None:
            torch.random.manual_seed(5)
            rand_h_2 = torch.randint(output_dim, size=(self.input_dim2,))
        if rand_s_2 is None:
            torch.random.manual_seed(7)
            rand_s_2 = 2 * torch.randint(2, size=(self.input_dim2,)) - 1

        self.sparse_sketch_matrix2 = self.generate_sketch_matrix(
            rand_h_2, rand_s_2, self.output_dim)

        self.sparse_sketch_matrix1 = self.sparse_sketch_matrix1.to(self.device)
        self.sparse_sketch_matrix2 = self.sparse_sketch_matrix2.to(self.device)

    def forward(self, bottom1, bottom2):
        """
        bottom1: 1st input, 4D Tensor of shape [batch_size, input_dim1, height, width].
        bottom2: 2nd input, 4D Tensor of shape [batch_size, input_dim2, height, width].
        """
        assert bottom1.size(1) == self.input_dim1 and \
            bottom2.size(1) == self.input_dim2

        batch_size, _, height, width = bottom1.size()

        bottom1_flat = bottom1.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim1).to(self.device)
        bottom2_flat = bottom2.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim2).to(self.device)

        sketch_1 = bottom1_flat.mm(self.sparse_sketch_matrix1)
        sketch_2 = bottom2_flat.mm(self.sparse_sketch_matrix2)

        # Build real+imag arrays to compute FFT, with imag = 0
        sketch_1 = torch.stack((sketch_1, torch.zeros(sketch_1.shape).to(device)), dim=-1)
        sketch_2 = torch.stack((sketch_2, torch.zeros(sketch_2.shape).to(device)), dim=-1)

        fft1 = torch.fft(sketch_1, signal_ndim=1)
        fft2 = torch.fft(sketch_2, signal_ndim=1)
        del sketch_1, sketch_2

        # Element-wise complex product
        real1, imag1 = fft1.transpose(0, -1)
        real2, imag2 = fft2.transpose(0, -1)
        prod = torch.stack((real1 * real2 - imag1 * imag2,
            real1 * imag2 + imag1 * real2), dim=0).transpose(0, -1)
        del real1, real2, imag1, imag2

        cbp_flat = torch.ifft(prod, signal_ndim=1)[..., 0]

        cbp = cbp_flat.view(batch_size, height, width, self.output_dim)

        if self.sum_pool:
            cbp = cbp.sum(dim=1).sum(dim=1)

        return cbp

    @staticmethod
    def generate_sketch_matrix(rand_h, rand_s, output_dim):
        """
        Return a sparse matrix used for tensor sketch operation in compact bilinear
        pooling
        Args:
            rand_h: an 1D tensor containing indices in interval `[0, output_dim)`.
            rand_s: an 1D tensor array of 1 and -1, having the same shape as `rand_h`.
            output_dim: the output dimensions of compact bilinear pooling.
        Returns:
            a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
        """

        # Generate a sparse matrix for tensor count sketch
        rand_h = rand_h.long()
        rand_s = rand_s.float()
        assert(rand_h.dim() == 1 and rand_s.dim() == 1 and len(rand_h) == len(rand_s))
        assert((rand_h >= 0).all() and (rand_h < output_dim).all())

        input_dim = len(rand_h)
        indices = torch.stack((torch.arange(input_dim, dtype=torch.long), rand_h))

        sparse_sketch_matrix = torch.sparse.FloatTensor(
            indices, rand_s, torch.Size([input_dim, output_dim]))

        return sparse_sketch_matrix.to_dense()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bottom1 = torch.randn(128, 512, 14, 14).to(device)
    bottom2 = torch.randn(128, 512, 14, 14).to(device)

    layer = CompactBilinearPooling(512, 512, 8000).to(device)

    out = layer(bottom1, bottom2)
