import os
import pathlib
import json
import time


def on_cluster():
    """
    :return: True if running job on cluster
    """
    p = pathlib.Path().absolute()
    id = p.parts[:3][-1]
    if id == 'users':
        return True
    else:
        return False

def get_timestamp():
    formatted_time = time.strftime('%d-%b-%y||%H:%M:%S')
    return formatted_time

def get_log_root():
    directory = f'{get_top_dir()}/images/logs'
    os.makedirs(directory, exist_ok=True)
    return directory

def get_checkpoint_root():
    directory = f'{get_top_dir()}/images/checkpoints'
    os.makedirs(directory, exist_ok=True)
    return directory

def get_data_root():
    return f'{get_top_dir()}/surVAE/data/downloads'


def get_top_dir():
    p = pathlib.Path().absolute()
    id = p.parts[:3][-1]
    if id == 'samklein':
        sv_ims = '/Users/samklein/PycharmProjects/surVAEsearcher'
    elif id == 'users':
        sv_ims = '/home/users/k/kleins/MLproject/surVAE'
    else:
        raise ValueError('Unknown path for saving images {}'.format(p))
    return sv_ims

def get_data_root():
    return f'{get_top_dir()}/surVAE/data/downloads'

def get_image_data_root(name):
    return f'{get_data_root()}/images/{name}'

class save_object():

    def __init__(self, directory, exp_name=None, args=None):
        self.image_dir = f'{get_top_dir()}/images/{directory}'
        self.exp_name = exp_name
        self.json_info = f"{self.image_dir}/exp_info_{exp_name}.json"
        os.makedirs(self.image_dir, exist_ok=True)
        if args is not None:
            self.register_experiment(args)


    def save_name(self, name, directory=None, extension='png'):
        if directory is None:
            image_dir = self.image_dir
            exp_name = '_' + self.exp_name
        else:
            image_dir = f'{get_top_dir()}/images/{directory}'
            exp_name = ''
            os.makedirs(image_dir, exist_ok=True)
        return f'{image_dir}/{name}{exp_name}.{extension}'


    def register_experiment(self, args):
        log_dict = vars(args)
        json_dict = json.dumps(log_dict)
        with open(self.json_info, "w") as file_name:
            json.dump(json_dict, file_name)

    def read_experiment(self, json_info):
        with open(json_info, "r") as file_name:
            json_dict = json.load(file_name)
        dict = json.loads(json_dict)
        self.exp_name = dict['outputname']
        return dict


# def make_splits(data):

