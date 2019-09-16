import pathlib
from copy import deepcopy, copy
import abc
import os

import pprint

import numpy as np
import cv2

import h5py

class BigBIRDObjectData:
    @staticmethod
    def _data_filename(sensor_id, image_id):
        return 'NP{}_{}'.format(sensor_id, image_id)

    class BoundingBox:
        def __init__(self, left, top, right, bottom):
            for attr in ('left', 'top', 'right', 'bottom'):
                setattr(self, attr, locals()[attr])

        @property
        def height(self):
            return np.abs(self.top-self.bottom) + 1

        @property
        def width(self):
            return np.abs(self.right-self.left) + 1

        @property
        def center(self):
            return np.array(((self.right+self.left)/2, (self.top+self.bottom)/2))

        @property
        def y_up(self):
            return self.top > self.bottom

        def resize_(self, new_width=None, new_height=None):
            cx, cy = self.center

            if new_width is not None:
                self.left, self.right = int(np.floor(cx - (new_width-1) / 2)), int(np.floor(cx + (new_width-1) / 2))

            if new_height is not None:
                new_coords = int(np.floor(cy - (new_height-1) / 2)), int(np.floor(cy + (new_height-1) / 2))
                self.top, self.bottom = new_coords[::-1] if self.y_up else new_coords

            return self

        def resize(self, *args, **kwargs):
            return deepcopy(self).resize_(*args, **kwargs)

        # def __copy__(self):
        #     return type(self)(**self.__dict__)

    def __len__(self):
        return self.num_images

    @staticmethod
    def _determine_size_to_crop(bbox_wh, crop_dims_wh=None, crop_dims_ratio=None):
        assert (crop_dims_wh is not None) ^ (crop_dims_ratio is not None)
        bbox_width, bbox_height = bbox_wh
        if crop_dims_wh is None:
            if 1 / bbox_width > crop_dims_ratio / bbox_height:
                new_crop_dims = (int(bbox_height / crop_dims_ratio), bbox_height)
            else:
                new_crop_dims = (bbox_width, int(bbox_width * crop_dims_ratio))
        else:
            ratio = min(crop_dims_wh / bbox_wh)
            new_crop_dims = (np.array(crop_dims_wh) / ratio).astype(int)
        return new_crop_dims

    def __init__(self, data_dir):
        self.data_dir = pathlib.Path(data_dir)
        all_images = list(self.data_dir.glob('NP*.jpg'))
        self.num_sensors = max([int(im.name[2:3]) for im in all_images])
        self.num_images = len(all_images) // self.num_sensors
        self.__crop_dims = None
        self.__dims_ratio = 1
        self.__preprocess_fcn = lambda x: x

        assert self.num_images*self.num_sensors == len(all_images), 'Different image counts for sensors in directory \'{}\''.format(self.data_dir)

        class __loader:
            def __init__(self, owner, sensor_id, **kwargs):
                self.owner = owner
                self.sensor_id = sensor_id
                self.__dict__.update(kwargs)

            def __iter__(self):
                for ii in range(len(self)):
                    yield self[ii]

            def __getitem__(self, item):
                loaded = self.load_item(item)
                return self.owner.preprocess_fcn(loaded) if isinstance(loaded, np.ndarray) else loaded

            def __len__(self):
                return len(self.owner)

            @abc.abstractmethod
            def load_item(self, item):
                pass

        class __image_loader(__loader):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def load_item(self, item):
                return cv2.imread(str(self.owner.data_dir / (BigBIRDObjectData._data_filename(self.sensor_id+1, 3*item) + '.jpg')))

        class __masks_loader(__loader):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def load_item(self, item):
                return cv2.imread(str(self.owner.data_dir / 'masks' / (BigBIRDObjectData._data_filename(self.sensor_id+1, 3*item) + '_mask.pbm')))

        class __bounding_box(__loader):
            def __init__(self, *args, padding=5, **kwargs):
                super().__init__(*args, **kwargs)
                self.padding = padding

            def load_item(self, item):
                mask = (self.owner.mask[self.sensor_id].load_item(item))[..., -1]
                row, col = np.where(mask == 0)
                return self.owner.BoundingBox(min(col)-self.padding, min(row)-self.padding, max(col)+self.padding, max(row)+self.padding)

        class __cropper(__loader):
            def __init__(self, *args, img_src, **kwargs):
                super().__init__(*args, **kwargs)

                self.img_src = img_src

            def load_item(self, item):
                bbox = self.owner.bbox[self.sensor_id].load_item(item)
                bbox_wh = np.array((bbox.width, bbox.height))

                crop_dims_wh = self.owner.crop_dims
                dims_ratio = self.owner.dims_ratio

                bbox.resize_(*self.owner._determine_size_to_crop(bbox_wh=bbox_wh, crop_dims_wh=crop_dims_wh, crop_dims_ratio=dims_ratio if crop_dims_wh is None else None))

                im = self.img_src[self.sensor_id].load_item(item)
                cropped_im = im[bbox.top:bbox.bottom, bbox.left:bbox.right, ...]
                return cropped_im if crop_dims_wh is None else \
                    cv2.resize(cropped_im, (0, 0), fx=crop_dims_wh[0]/(bbox.width-1), fy=crop_dims_wh[1]/(bbox.height-1))


                # return self.owner.BoundingBox(min(col)-self.padding, min(row)-self.padding, max(col)+self.padding, max(row)+self.padding)

        class __poses(__loader):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def load_item(self, item):
                fname = self.owner.data_dir / 'poses' / (BigBIRDObjectData._data_filename(self.sensor_id+1, 3*item) + '_pose.h5')
                fname.stat()    # access to make sure exists
                return self.owner._hdf5_to_numpy_adapter(h5py.File(fname))

        self.rgb = [__image_loader(owner=self, sensor_id=id) for id in range(self.num_sensors)]
        self.mask = [__masks_loader(owner=self, sensor_id=id) for id in range(self.num_sensors)]
        self.bbox = [__bounding_box(owner=self, sensor_id=id) for id in range(self.num_sensors)]
        self.cropped_rgb = [__cropper(owner=self, sensor_id=id, img_src=self.rgb) for id in range(self.num_sensors)]
        self.cropped_mask = [__cropper(owner=self, sensor_id=id, img_src=self.mask) for id in range(self.num_sensors)]
        self.poses = __poses(owner=self, sensor_id=4, img_src=self.mask)

    @property
    def crop_dims(self):
        return self.__crop_dims

    @crop_dims.setter
    def crop_dims(self, dims):
        self.__crop_dims = dims

    @property
    def dims_ratio(self):
        return self.__dims_ratio

    @dims_ratio.setter
    def dims_ratio(self, dims):
        self.__dims_ratio = dims

    @property
    def preprocess_fcn(self):
        return self.__preprocess_fcn

    @preprocess_fcn.setter
    def preprocess_fcn(self, fcn):
        assert fcn is None or callable(fcn)
        self.__preprocess_fcn = (lambda x: x) if fcn is None else fcn

    class _hdf5_to_numpy_adapter:
        def __init__(self, obj):
            self.obj = obj

        def __getitem__(self, item):
            return np.array(self.obj[item])

        def __repr__(self):
            return object.__repr__(self) + ' wrapping: ' + self.obj.__repr__() + '\n' + pprint.pformat(list(self.obj.keys()))

        def __getattr__(self, item):
            return getattr(self.obj, item)

    @property
    def calibration(self):
        return self._hdf5_to_numpy_adapter(h5py.File(self.data_dir / 'calibration.h5'))



class BigBIRDDataset:
    def __init__(self, data_root):
        self.data_root = pathlib.Path(data_root)

    def __repr__(self):
        return object.__repr__(self) + '\nObjects: \n' + pprint.pformat(os.listdir(self.data_root))

    def __getitem__(self, item):
        return BigBIRDObjectData(self.data_root / item)