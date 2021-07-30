from nflows import transforms


def get_transform(inp_dim=1, nodes=64, num_blocks=2, nstack=2, tails='linear', tail_bound=4, num_bins=10,
                  context_features=1):
    transform_list = []
    for i in range(nstack):
        transform_list += [
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(inp_dim, nodes,
                                                                               num_blocks=num_blocks,
                                                                               tail_bound=tail_bound, num_bins=num_bins,
                                                                               tails=tails,
                                                                               context_features=context_features)]

        transform_list += [transforms.ReversePermutation(inp_dim)]

    return transforms.CompositeTransform(transform_list[:-1])
