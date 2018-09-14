import argparse



def restricted_float(x=None):
    if x is not None:
        x = float(x)
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


def add_suffix(path, suffix):
    list_path = path.split('/')
    fname = list_path[-1]
    list_fname = fname.split('.')
    list_fname[0]+= suffix
    fname_suffix = '.'.join(list_fname)
    list_path[-1] = fname_suffix
    path_suffix = '/'.join(list_path)
    return path_suffix


class Subject():
    def __init__(self, path='', fname_im='', fname_mask='', ori='', type_set=None, im_data=None, mask_data=None, hdr=None, im_data_preproc=None, im_mask_preproc=None):
        self.path = path
        self.fname_im = fname_im
        self.fname_mask = fname_mask
        #
        self.type = type_set
        #
        self.im_data = im_data
        self.mask_data = mask_data
        self.hdr = None
        self.orientation = ori
        #
        self.im_data_preprocessed = im_data_preproc
        self.im_mask_preprocessed = im_mask_preproc
    #
    def __repr__(self):
        to_print = '\nSubject:   '
        to_print += '   path: '+self.path
        to_print += '   fname image: ' + self.fname_im
        to_print += '   fname mask: ' + self.fname_mask
        to_print += '   orientation of im: ' + self.orientation
        to_print += '   used for : ' + str(self.type)
        return to_print
