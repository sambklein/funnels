import os
import pathlib
import json


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

class save_object():

    def __init__(self, directory, exp_name=None, args=None):
        self.image_dir = f'{get_top_dir()}/images/{directory}'
        self.exp_name = exp_name
        self.json_info = f"{self.image_dir}/{exp_name}_exp_info.json"
        os.makedirs(self.image_dir, exist_ok=True)
        if args is not None:
            self.register_experiment(args)


    def save_name(self, name, directory=None, extension='png'):
        if directory is None:
            image_dir = self.image_dir
            exp_name = self.exp_name + '_'
        else:
            image_dir = f'{get_top_dir()}/images/{directory}'
            exp_name = ''
            os.makedirs(image_dir, exist_ok=True)
        return f'{image_dir}/{exp_name}{name}.{extension}'


    def register_experiment(self, args):
        log_dict = vars(args)
        json_dict = json.dumps(log_dict)
        with open(self.json_info, "w") as file_name:
            json.dump(json_dict, file_name)


    def read_experiment(self, args):
        with open(self.json_info, "w") as file_name:
            json_dict = json.load(file_name)
        return json.loads(json_dict)
