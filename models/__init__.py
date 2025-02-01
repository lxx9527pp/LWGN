def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'lwgn':
        from .model import LWGN
        return LWGN
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
