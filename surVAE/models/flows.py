from nflows import transforms


def get_transform(inp_dim=1, nodes=64, num_blocks=2, nstack=2, tails=None, tail_bound=1., num_bins=10,
                  context_features=1, lu=1):
    transform_list = []
    for i in range(nstack):

        if tails:
            tb = tail_bound
        else:
            tb = tail_bound if i == 0 else None

        transform_list += [
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(inp_dim, nodes,
                                                                               num_blocks=num_blocks,
                                                                               tail_bound=tb, num_bins=num_bins,
                                                                               tails=tails,
                                                                               context_features=context_features)]
        if (tails is None) and (tail_bound is not None) and (i == nstack - 1):
            transform_list += [transforms.standard.PointwiseAffineTransform(-tail_bound, 2 * tail_bound)]

        if lu:
            transform_list += [transforms.LULinear(inp_dim)]
        else:
            transform_list += [transforms.ReversePermutation(inp_dim)]

    return transforms.CompositeTransform(transform_list[:-1])
