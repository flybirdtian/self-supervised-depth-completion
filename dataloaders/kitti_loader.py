import os
import os.path
import glob
import fnmatch # pattern matching
import numpy as np
from numpy import linalg as LA
from  random import choice
from PIL import Image
import torch
import torch.utils.data as data
import cv2
from dataloaders import transforms
from dataloaders.pose_estimator import get_pose_pnp

input_options = ['d', 'rgb', 'rgbd', 'g', 'gd']

def load_calib():
    """
    Temporarily hardcoding the calibration matrix using calib file from 2011_09_26
    """
    calib = open("dataloaders/calib_cam_to_cam.txt", "r")
    lines = calib.readlines()
    P_rect_line = lines[25]

    Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
    Proj = np.reshape(np.array([float(p) for p in Proj_str]),(3,4)).astype(np.float32)
    K = Proj[:3,:3] # camera matrix

    # note: we will take the center crop of the images during augmentation
    # that changes the optical centers, but not focal lengths
    K[0,2] = K[0,2] - 13 # from width = 1242 to 1216, with a 13-pixel cut on both sides
    K[1,2] = K[1,2] - 11.5 # from width = 375 to 352, with a 11.5-pixel cut on both sides
    return K

root_d = os.path.join('..', 'data', 'kitti_depth')
root_rgb = os.path.join('..', 'data', 'kitti_rgb')
def get_paths_and_transform(split, args):
    assert (args.use_d or args.use_rgb or args.use_g), 'no proper input selected'

    if split == "train":
        transform = train_transform
        glob_gt = "train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png"
        pattern_d = ("groundtruth","velodyne_raw")
        def get_rgb_paths(p):
          ps = p.split('/')
          pnew = '/'.join([root_rgb]+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
          return pnew
    elif split == "val":
        if args.val == "full":
            transform = val_transform
            glob_gt = "val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png"
            pattern_d = ("groundtruth","velodyne_raw")
            def get_rgb_paths(p):
              ps = p.split('/')
              pnew = '/'.join([root_rgb]+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
              return pnew
        elif args.val == "select":
            transform = no_transform
            glob_gt = "val_selection_cropped/groundtruth_depth/*.png"
            pattern_d = ("groundtruth_depth","velodyne_raw")
            def get_rgb_paths(p):
              return p.replace("groundtruth_depth","image")
    elif split == "test_completion":
        transform = no_transform
        glob_gt  = None #"test_depth_completion_anonymous/"
        base = "/test_depth_completion_anonymous/"
        glob_d   = root_d+base+"/velodyne_raw/*.png"
        glob_rgb = root_d+base+"/image/*.png"
    elif split == "test_prediction":
        transform = no_transform
        glob_gt  = None #"test_depth_completion_anonymous/"
        base = "/test_depth_prediction_anonymous/"
        glob_d   = root_d+base+"/velodyne_raw/*.png"
        glob_rgb = root_d+base+"/image/*.png"
    else:
        raise ValueError("Unrecognized split "+str(split))

    if glob_gt is not None:
        glob_gt = os.path.join(root_d,glob_gt)
        paths_gt = sorted(glob.glob(glob_gt))
        paths_d = [p.replace(pattern_d[0],pattern_d[1]) for p in paths_gt]
        paths_rgb = [get_rgb_paths(p) for p in paths_gt]
    else: # test and only has d or rgb
        paths_rgb = sorted(glob.glob(glob_rgb))
        paths_gt = [None]*len(paths_rgb)
        if split == "test_prediction":
            paths_d = [None]*len(paths_rgb) # test_prediction has no sparse depth
        else:
            paths_d = sorted(glob.glob(glob_d))

    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0:
        raise(RuntimeError("Found 0 images in data folders"))
    if len(paths_d) == 0 and args.use_d:
        raise(RuntimeError("Requested sparse depth but none was found"))
    if len(paths_rgb) == 0 and args.use_rgb:
        raise(RuntimeError("Requested rgb images but none was found"))
    if len(paths_rgb) == 0 and args.use_g:
        raise(RuntimeError("Requested gray images but no rgb was found"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
        raise(RuntimeError("Produced different sizes for datasets"))

    paths = {"rgb":paths_rgb, "d":paths_d, "gt":paths_gt}
    return paths, transform

def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8') # in the range [0,255]
    img_file.close()
    return rgb_png

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth,-1)
    return depth

oheight, owidth = 352, 1216

def drop_depth_measurements(depth, prob_keep):
    mask = np.random.binomial(1, prob_keep, depth.shape)
    depth *= mask
    return depth

def train_transform(rgb, sparse, target, rgb_near, args):
    # s = np.random.uniform(1.0, 1.5) # random scaling
    # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

    transform_geometric = transforms.Compose([
        # transforms.Rotate(angle),
        # transforms.Resize(s),
        transforms.BottomCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)
    ])
    if sparse is not None:
        sparse = transform_geometric(sparse)
    target = transform_geometric(target)
    if rgb is not None:
        brightness = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        contrast = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        saturation = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        transform_rgb = transforms.Compose([
            transforms.ColorJitter(brightness, contrast, saturation, 0),
            transform_geometric
        ])
        rgb = transform_rgb(rgb)
        if rgb_near is not None:
            rgb_near = transform_rgb(rgb_near)
    # sparse = drop_depth_measurements(sparse, 0.9)

    return rgb, sparse, target, rgb_near

def val_transform(rgb, sparse, target, rgb_near, args):
    transform = transforms.Compose([
        transforms.BottomCrop((oheight, owidth)),
    ])
    if rgb is not None:
        rgb = transform(rgb)
    if sparse is not None:
        sparse = transform(sparse)
    if target is not None:
        target = transform(target)
    if rgb_near is not None:
        rgb_near = transform(rgb_near)
    return rgb, sparse, target, rgb_near

def no_transform(rgb, sparse, target, rgb_near, args):
    return rgb, sparse, target, rgb_near

to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()

def handle_gray(rgb, args):
    if rgb is None:
        return None, None
    if not args.use_g:
        return rgb, None
    else:
        img = np.array(Image.fromarray(rgb).convert('L'))
        img = np.expand_dims(img,-1)
        if not args.use_rgb:
            rgb_ret = None
        else:
            rgb_ret = rgb
        return rgb_ret, img

def get_rgb_near(path, args):
    assert path is not None, "path is None"

    def extract_frame_id(filename):
        head, tail = os.path.split(filename)
        number_string = tail[0:tail.find('.')]
        number = int(number_string)
        return head, number

    def get_nearby_filename(filename, new_id):
        head, _ = os.path.split(filename)
        new_filename = os.path.join(head, '%010d.png' % new_id)
        return new_filename

    head, number = extract_frame_id(path)
    count = 0
    max_frame_diff = 3
    candidates = [i-max_frame_diff for i in range(max_frame_diff*2+1) if i-max_frame_diff!=0]
    while True:
        random_offset = choice(candidates)
        path_near = get_nearby_filename(path, number+random_offset)
        if os.path.exists(path_near):
            break
        assert count < 20, "cannot find a nearby frame in 20 trials for {}".format(path_rgb_tgt)

    return rgb_read(path_near)

class KittiDepth(data.Dataset):
    """A data loader for the Kitti dataset
    """
    def __init__(self, split, args):
        self.args = args
        self.split = split
        paths, transform = get_paths_and_transform(split, args)
        self.paths = paths
        self.transform = transform
        self.K = load_calib()
        self.threshold_translation = 0.1

    def __getraw__(self, index):
        rgb = rgb_read(self.paths['rgb'][index]) if \
            (self.paths['rgb'][index] is not None and (self.args.use_rgb or self.args.use_g)) else None
        sparse = depth_read(self.paths['d'][index]) if \
            (self.paths['d'][index] is not None and self.args.use_d) else None
        target = depth_read(self.paths['gt'][index]) if \
            self.paths['gt'][index] is not None else None
        rgb_near = get_rgb_near(self.paths['rgb'][index], self.args) if \
            self.split == 'train' and self.args.use_pose else None
        return rgb, sparse, target, rgb_near

    def __getitem__(self, index):
        rgb, sparse, target, rgb_near = self.__getraw__(index)
        rgb, sparse, target, rgb_near = self.transform(rgb,sparse, target, rgb_near, self.args)
        r_mat, t_vec = None, None
        if self.split == 'train' and self.args.use_pose:
            success, r_vec, t_vec = get_pose_pnp(rgb, rgb_near, sparse, self.K)
            # discard if translation is too small
            success = success and LA.norm(t_vec) > self.threshold_translation
            if success:
                r_mat, _ = cv2.Rodrigues(r_vec)
            else:
                # return the same image and no motion when PnP fails
                rgb_near = rgb
                t_vec = np.zeros((3,1))
                r_mat = np.eye(3)

        rgb, gray = handle_gray(rgb, self.args)
        candidates = {"rgb":rgb, "d":sparse, "gt":target, \
            "g":gray, "r_mat":r_mat, "t_vec":t_vec, "rgb_near":rgb_near}
        items = {key:to_float_tensor(val) for key, val in candidates.items() if val is not None}

        return items

    def __len__(self):
        return len(self.paths['gt'])


def get_dataset_path(base_dir, setname='train'):
    """
    get dataset path according to setname
    :param base_dir: basic data dir
    :param setname: train, val, seval, test
    :return: lidar_dir, depth_dir, rgb_dir
    """

    import os
    if setname == 'train':
        lidar_dir = os.path.join(base_dir, 'data_depth_velodyne', 'train')
        depth_dir = os.path.join(base_dir, 'data_depth_annotated', 'train')
        rgb_dir = os.path.join(base_dir, 'raw')
    elif setname == 'val':
        lidar_dir = os.path.join(base_dir, 'data_depth_velodyne', 'val')
        depth_dir = os.path.join(base_dir, 'data_depth_annotated', 'val')
        rgb_dir = os.path.join(base_dir, 'raw')
    elif setname == 'selval':
        lidar_dir = os.path.join(base_dir, 'val_selection_cropped', 'velodyne_raw')
        depth_dir = os.path.join(base_dir, 'val_selection_cropped', 'groundtruth_depth')
        rgb_dir = os.path.join(base_dir, 'val_selection_cropped', 'image')
    elif setname == 'test':
        lidar_dir = os.path.join(base_dir, 'test_depth_completion_anonymous', 'velodyne_raw')
        depth_dir = os.path.join(base_dir, 'test_depth_completion_anonymous', 'velodyne_raw')
        rgb_dir = os.path.join(base_dir, 'test_depth_completion_anonymous', 'image')
    else:
        raise ValueError("Unrecognized setname "+str(setname))

    return lidar_dir, depth_dir, rgb_dir


def get_transform(mode):
    if mode == 'train':
        transform = train_transform
    elif mode == 'eval':
        transform = val_transform
    else:
        raise ValueError("Unrecognized mode "+str(mode))

    return transform


class KittiDataset(data.Dataset):
    """
    origin format as original
    """
    def __init__(self, base_dir, mode, setname, args):
        """
        :param base_dir:
        :param setname:
        :param transform:
        :param return_idx:
        :param crop:
        """
        self.dataset_name = 'kitti'

        self.mode = mode
        self.setname = setname
        self.args = args

        self.base_dir = base_dir
        lidar_dir, depth_dir, rgb_dir = self.get_paths(base_dir)
        self.lidar_dir = lidar_dir
        self.depth_dir = depth_dir
        self.rgb_dir = rgb_dir

        self.transform = get_transform(mode)

        self.crop_h = 352
        self.crop_w = 1216

        # sparsity
        self.use_sparsity = False
        self.sparsity_ratio = 0.2

        self.crop = True

        self.lidars = list(sorted(glob.iglob(self.lidar_dir + "/**/*.png", recursive=True)))

    def __len__(self):
        return len(self.lidars)

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        return self.getitem(index)

    def get_file_name(self, item):
        lidar_path = self.lidars[item]
        file_names = lidar_path.split('/')
        return file_names[-1]

    def get_paths(self, base_dir):
        return get_dataset_path(base_dir, setname=self.setname)

    def split_selval_filename(self, filename):
        mid = 'velodyne_raw_'
        name_split = filename.split(mid)
        pre, pos = name_split[0], name_split[1]
        return pre, mid, pos

    def get_depth_path(self, lidar_path):
        depth_path = None
        if self.setname == 'train' or self.setname == 'val':
            file_names = lidar_path.split('/')
            depth_path = os.path.join(self.depth_dir, *file_names[-5:-3], 'groundtruth', *file_names[-2:])
        elif self.setname == 'test':
            file_names = lidar_path.split('/')
            depth_path = os.path.join(self.depth_dir, file_names[-1])
        elif self.setname == 'selval':
            file_names = lidar_path.split('/')
            file_name = file_names[-1]
            pre, mid, pos = self.split_selval_filename(file_name)
            depth_file_name = pre + "groundtruth_depth_" + pos
            depth_path = os.path.join(self.depth_dir, depth_file_name)
        return depth_path

    def get_rgb_path(self, lidar_path):
        rgb_path = None
        if self.setname == 'train' or self.setname == 'val':
            file_names = lidar_path.split('/')
            rgb_path = os.path.join(self.rgb_dir, file_names[-5].split('_drive')[0], file_names[-5],
                                    file_names[-2], 'data', file_names[-1])
        elif self.setname == 'test':
            file_names = lidar_path.split('/')
            rgb_path = os.path.join(self.rgb_dir, file_names[-1])
        elif self.setname == 'selval':
            file_names = lidar_path.split('/')
            file_name = file_names[-1]
            pre, mid, pos = self.split_selval_filename(file_name)
            rgb_file_name = pre + "image_" + pos
            rgb_path = os.path.join(self.rgb_dir, rgb_file_name)
        return rgb_path

    def get_raw_data(self, lidar_path):
        depth_path = self.get_depth_path(lidar_path)
        rgb_path = self.get_rgb_path(lidar_path)
        assert(rgb_path != '')

        depth = self.read_depth(depth_path)
        lidar = self.read_depth(lidar_path)
        rgb = self.read_rgb(rgb_path)

        if self.use_sparsity:
            rand_value = np.random.rand(*lidar.shape)
            rand_mask = rand_value <= self.sparsity_ratio
            lidar = lidar * rand_mask

        return rgb, lidar, depth

    def crop_data(self, data):
        w = data.shape[1]
        lp = (w - self.crop_w) // 2
        return data[-self.crop_h:, lp:lp + self.crop_w, :]

    def deal_data(self, rgb, lidar, depth):
        depth = np.expand_dims(depth, axis=2)
        lidar = np.expand_dims(lidar, axis=2)

        if self.transform:
            rgb, lidar, depth, rgb_ = self.transform(rgb, lidar, depth, rgb, self.args)

        gray = np.array(Image.fromarray(rgb).convert('L'))
        gray = np.expand_dims(gray, -1)

        if self.crop:
            rgb = self.crop_data(rgb)
            lidar = self.crop_data(lidar)
            depth = self.crop_data(depth)
            gray = self.crop_data(gray)

        return rgb.astype(np.float32), lidar.astype(np.float32), depth.astype(np.float32), gray.astype(np.float32)

    def get_ori_item(self, index):
        lidar_path = self.lidars[index]
        rgb, lidar, depth = self.get_raw_data(lidar_path)

        out = (rgb, lidar, depth)

        return out

    def get_ori_shape(self, index):
        lidar_path = self.lidars[index]
        return self._get_ori_shape(lidar_path)

    def _get_ori_shape(self, lidar_path):
        import magic
        import re

        info = magic.from_file(lidar_path)
        size_info = re.search('(\d+) x (\d+)', info).groups()
        width = int(size_info[0])
        height = int(size_info[1])
        return height, width

    def getitem(self, index):
        lidar_path = self.lidars[index]
        rgb, lidar, depth = self.get_raw_data(lidar_path)

        # verify shape size
        ori_shape = lidar.shape
        assert ori_shape == depth.shape == rgb.shape[0:2] == self.get_ori_shape(index)

        rgb, lidar, depth, gray = self.deal_data(rgb, lidar, depth)

        index_3d = np.zeros((1, 1, 1))
        index_3d[0, 0, 0] = index

        candidates = {"rgb":rgb, "d":lidar, "gt":depth, "g":gray, "index":index_3d, "r_mat":None, "t_vec":None, "rgb_near":None}
        items = {key:to_float_tensor(val) for key, val in candidates.items() if val is not None}

        return items

    def read_rgb(self, path):
        img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
        return img

    def read_depth(self, path):
        depth_png = np.array(Image.open(path), dtype=np.uint16)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert (np.max(depth_png) > 255)
        depth_image = (depth_png / 256.).astype(np.float32)
        # depth_image = self.mask_depth_image(depth_image, self.depth_start, self.depth_end)
        return depth_image


class VKittiDataset(data.Dataset):
    """
    origin format as original
    """
    def __init__(self, base_dir, mode, args, setname='train', subset='clone'):
        """
        :param base_dir:
        :param setname:
        :param transform:
        :param return_seg:
        :param return_idx:
        :param use_stereo:
        :param crop:
        :param compose_depth:
        :param split_depth: True or False, if True, split depth into multiple channels according to depth
        :param split_mode: single_offset or multiple_offset, output single channel offset or multiple channel offset
        """
        self.mode = mode
        self.setname = setname
        self.subset = subset
        self.args = args

        self.base_dir = base_dir
        lidar_dir, depth_dir, rgb_dir = self.get_paths(base_dir)
        self.lidar_dir = lidar_dir
        self.depth_dir = depth_dir
        self.rgb_dir = rgb_dir

        self.transform = get_transform(mode)

        self.crop_h = 352
        self.crop_w = 1216

        # sparsity
        self.use_sparsity = False
        self.sparsity_ratio = 0.2

        self.crop = True

        self.lidars = list(sorted(glob.iglob(self.lidar_dir + "/**/" + subset + "/*.png", recursive=True)))

    def __len__(self):
        return len(self.lidars)

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        return self.getitem(index)

    def get_file_name(self, item):
        lidar_path = self.lidars[item]
        file_names = lidar_path.split('/')
        return file_names[-1]

    def get_paths(self, base_dir):
        """
        get dataset path according to setname
        :param base_dir: basic data dir
        :param setname: train, val, seval, test
        :param use_compose_depth: use left-right composed depth as ground-truth or not
        :return: lidar_dir, depth_dir, rgb_dir
        """
        if self.setname == 'train':
            lidar_dir = os.path.join(base_dir, 'train', 'sparse_lidar')
            depth_dir = os.path.join(base_dir, 'train', 'ka_depth')
            rgb_dir = os.path.join(base_dir, 'train', 'rgb')
        elif self.setname == 'test':
            lidar_dir = os.path.join(base_dir, 'test', 'sparse_lidar')
            depth_dir = os.path.join(base_dir, 'test', 'ka_depth')
            rgb_dir = os.path.join(base_dir, 'test', 'rgb')
        else:
            raise Exception("setname {} does not exist!".format(self.setname))

        return lidar_dir, depth_dir, rgb_dir

    def get_rgb_path(self, lidar_path):
        file_names = lidar_path.split('/')
        rgb_path = os.path.join(self.rgb_dir, *file_names[-3:])
        return rgb_path

    def get_depth_path(self, lidar_path):
        file_names = lidar_path.split('/')
        depth_path = os.path.join(self.depth_dir, *file_names[-3:])
        return depth_path

    def get_raw_data(self, lidar_path):
        depth_path = self.get_depth_path(lidar_path)
        rgb_path = self.get_rgb_path(lidar_path)
        assert(rgb_path != '')

        depth = self.read_depth(depth_path)
        lidar = self.read_depth(lidar_path)
        rgb = self.read_rgb(rgb_path)

        if self.use_sparsity:
            rand_value = np.random.rand(*lidar.shape)
            rand_mask = rand_value <= self.sparsity_ratio
            lidar = lidar * rand_mask

        return rgb, lidar, depth

    def crop_data(self, data):
        w = data.shape[1]
        lp = (w - self.crop_w) // 2
        return data[-self.crop_h:, lp:lp + self.crop_w, :]

    def deal_data(self, rgb, lidar, depth):
        depth = np.expand_dims(depth, axis=2)
        lidar = np.expand_dims(lidar, axis=2)

        if self.transform:
            rgb, lidar, depth, rgb_ = self.transform(rgb, lidar, depth, rgb, self.args)

        gray = np.array(Image.fromarray(rgb).convert('L'))
        gray = np.expand_dims(gray, -1)

        # padding to size(input_h, input_w) at the bottom and right of the image
        if self.crop:
            rgb = self.crop_data(rgb)
            lidar = self.crop_data(lidar)
            depth = self.crop_data(depth)
            gray = self.crop_data(gray)

        return rgb.astype(np.float32), lidar.astype(np.float32), depth.astype(np.float32), gray.astype(np.float32)

    def get_ori_item(self, index):
        lidar_path = self.lidars[index]
        rgb, lidar, depth = self.get_raw_data(lidar_path)

        out = (rgb, lidar, depth)

        return out

    def get_ori_shape(self, index):
        lidar_path = self.lidars[index]
        return self._get_ori_shape(lidar_path)

    def _get_ori_shape(self, lidar_path):
        import magic
        import re

        info = magic.from_file(lidar_path)
        size_info = re.search('(\d+) x (\d+)', info).groups()
        width = int(size_info[0])
        height = int(size_info[1])
        return height, width

    def getitem(self, index):
        lidar_path = self.lidars[index]
        rgb, lidar, depth = self.get_raw_data(lidar_path)

        # verify shape size
        ori_shape = lidar.shape
        assert ori_shape == depth.shape == rgb.shape[0:2] == self.get_ori_shape(index)

        rgb, lidar, depth, gray = self.deal_data(rgb, lidar, depth)

        index_3d = np.zeros((1, 1, 1))
        index_3d[0, 0, 0] = index

        candidates = {"rgb":rgb, "d":lidar, "gt":depth, "g":gray, "index":index_3d, "r_mat":None, "t_vec":None, "rgb_near":None}
        items = {key:to_float_tensor(val) for key, val in candidates.items() if val is not None}

        return items

    def read_rgb(self, path):
        img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
        return img

    def read_depth(self, path):
        depth_png = np.array(Image.open(path), dtype=np.uint16)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert (np.max(depth_png) > 100)
        depth_image = (depth_png / 100.).astype(np.float32)
        # depth_image = self.mask_depth_image(depth_image, self.depth_start, self.depth_end)
        return depth_image


class OurDataset(data.Dataset):
    """
    origin format as original
    """
    def __init__(self, base_dir, mode, setname, args):
        """
        :param base_dir:
        :param setname:
        """
        self.base_dir = base_dir
        self.mode = mode
        self.setname = setname
        self.args = args

        lidar_dir, depth_dir, rgb_dir = self.get_paths(base_dir)
        self.lidar_dir = lidar_dir
        self.depth_dir = depth_dir
        self.rgb_dir = rgb_dir

        self.transform = get_transform(mode)

        self.crop_h = 352
        self.crop_w = 1216

        self.crop = True

        self.lidars = list(sorted(glob.iglob(self.lidar_dir + "/**/*.png", recursive=True)))

    def get_file_name(self, item):
        lidar_path = self.lidars[item]
        file_names = lidar_path.split('/')
        return file_names[-1]

    def __len__(self):
        return len(self.lidars)

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        return self.getitem(index)

    def get_paths(self, base_dir):
        lidar_dir = os.path.join(base_dir, 'lidar')
        depth_dir = os.path.join(base_dir, 'depth')
        rgb_dir = os.path.join(base_dir, 'image')

        return lidar_dir, depth_dir, rgb_dir

    def get_depth_path(self, lidar_path):
        file_names = lidar_path.split('/')
        depth_path = os.path.join(self.depth_dir, file_names[-1])
        return depth_path

    def get_rgb_path(self, lidar_path):
        file_names = lidar_path.split('/')
        rgb_path = os.path.join(self.rgb_dir, file_names[-1])
        # rgb_path = os.path.join(self.rgb_dir, file_names[-1][:-4] + ".jpg")
        return rgb_path

    def get_raw_data(self, lidar_path):
        depth_path = self.get_depth_path(lidar_path)
        rgb_path = self.get_rgb_path(lidar_path)
        assert(rgb_path != '')

        depth = self.read_depth(depth_path)
        lidar = self.read_depth(lidar_path)
        rgb = self.read_rgb(rgb_path)

        return rgb, lidar, depth

    def crop_data(self, data):
        w = data.shape[1]
        lp = (w - self.crop_w) // 2
        return data[-self.crop_h:, lp:lp + self.crop_w, :]

    def deal_data(self, rgb, lidar, depth):
        depth = np.expand_dims(depth, axis=2)
        lidar = np.expand_dims(lidar, axis=2)

        gray = np.array(Image.fromarray(rgb).convert('L'))
        gray = np.expand_dims(gray, -1)

        if self.transform:
            rgb, lidar, depth = self.transform(rgb, lidar, depth)

        # crop to size(input_h, input_w) at the bottom and right of the image
        if self.crop:
            rgb = self.crop_data(rgb)
            lidar = self.crop_data(lidar)
            depth = self.crop_data(depth)
            gray = self.crop_data(gray)

        return rgb.astype(np.float32), lidar.astype(np.float32), depth.astype(np.float32), gray.astype(np.float32)

    def get_ori_shape(self, index):
        lidar_path = self.lidars[index]
        return self._get_ori_shape(lidar_path)

    def _get_ori_shape(self, lidar_path):
        import magic
        import re

        info = magic.from_file(lidar_path)
        size_info = re.search('(\d+) x (\d+)', info).groups()
        width = int(size_info[0])
        height = int(size_info[1])
        return height, width

    def getitem(self, index):
        lidar_path = self.lidars[index]
        rgb, lidar, depth = self.get_raw_data(lidar_path)

        # verify shape size
        ori_shape = lidar.shape
        assert ori_shape == depth.shape == rgb.shape[0:2] == self.get_ori_shape(index)

        rgb, lidar, depth, gray = self.deal_data(rgb, lidar, depth)

        index_3d = np.zeros((1, 1, 1))
        index_3d[0, 0, 0] = index

        candidates = {"rgb":rgb, "d":lidar, "gt":depth, "g":gray, "index": index_3d, "r_mat":None, "t_vec":None, "rgb_near":None}
        items = {key:to_float_tensor(val) for key, val in candidates.items() if val is not None}

        return items

    def read_rgb(self, path):
        img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
        return img

    def read_depth(self, path):
        depth_png = np.array(Image.open(path), dtype=np.uint16)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert (np.max(depth_png) > 255)
        depth_image = (depth_png / 256.).astype(np.float32)
        # depth_image = self.mask_depth_image(depth_image, self.depth_start, self.depth_end)
        return depth_image


class NuScenesDataset(OurDataset):
    def __init__(self, base_dir, mode, setname, args):
        super().__init__(base_dir=base_dir, mode=mode, setname=setname, args=args)

    def get_paths(self, base_dir):
        lidar_dir = os.path.join(base_dir, 'lidar')
        depth_dir = os.path.join(base_dir, 'lidar_agg')
        rgb_dir = os.path.join(base_dir, 'image')

        return lidar_dir, depth_dir, rgb_dir

