from nflows.transforms import SqueezeTransform, Transform
import torch


class ReverseSqueezeTransform(SqueezeTransform):

    def get_output_shape(self, c, h, w):
        return (c // self.factor ** 2, h * self.factor, w * self.factor)

    def forward(self, inputs, context=None):
        return super(ReverseSqueezeTransform, self).inverse(inputs, context)

    def inverse(self, inputs, context=None):
        return super(ReverseSqueezeTransform, self).forward(inputs, context)


class PaddingSurjection(Transform):
    """
    This is a very specific kind of padding operation, essentially just implemented for one use case.
    """

    def __init__(self, pad=0):
        super(PaddingSurjection, self).__init__()
        self.register_buffer('pad', torch.tensor(pad, dtype=torch.int32))

    def get_ldj(self, inputs):
        return torch.zeros(inputs.shape[0])

    def forward(self, inputs, context=None):
        output = torch.nn.functional.pad(inputs, (0, self.pad, 0, self.pad))
        ldj = self.get_ldj(inputs)
        return output, ldj

    def inverse(self, inputs, context=None):
        return inputs[..., :-1, :-1], -self.get_ldj(inputs)

