from nflows.transforms import SqueezeTransform


class ReverseSqueezeTransform(SqueezeTransform):

    def get_output_shape(self, c, h, w):
        return (c // self.factor ** 2, h * self.factor, w * self.factor)

    def forward(self, inputs, context=None):
        return super(ReverseSqueezeTransform, self).inverse(inputs, context)

    def inverse(self, inputs, context=None):
        return super(ReverseSqueezeTransform, self).forward(inputs, context)