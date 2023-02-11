'''
Project: SelfSupSurg
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

import logging
import os
from fvcore.common.file_io import PathManager
from PIL import Image
import pickle
from torchvision.datasets import ImageFolder
from vissl.data.data_helper import QueueDataset, get_mean_image


class DiskImageDatasetSurgery(QueueDataset):
    def __init__(self, cfg, data_source, path, split, dataset_name):
        super(DiskImageDatasetSurgery, self).__init__(
            queue_size=cfg["DATA"][split]["BATCHSIZE_PER_REPLICA"]
        )
        assert data_source in [
            "disk_filelist_surgery"
        ], "data_source must be disk_filelist_surgery"

        self.cfg = cfg
        self.split = split
        self.dataset_name = dataset_name
        self.data_source = data_source
        self._path = path
        self.image_dataset = []
        self.image_ids = []
        self.is_initialized = False

        self._load_data(path)
        self._num_samples = len(self.image_dataset)
        self._remove_prefix = cfg["DATA"][self.split]["REMOVE_IMG_PATH_PREFIX"]
        self._new_prefix = cfg["DATA"][self.split]["NEW_IMG_PATH_PREFIX"]
        if self.data_source == "disk_filelist_surgery":
        # Set dataset to null so that workers dont need to pickle this file.
        # This saves memory when disk_filelist is large, especially when memory mapping.
           self.image_dataset = []
        # whether to use QueueDataset class to handle invalid images or not
        self.enable_queue_dataset = cfg["DATA"][self.split]["ENABLE_QUEUE_DATASET"]

    def _load_data(self, path):
        if self.data_source == "disk_filelist_surgery":
            root_dir = os.path.dirname(path.replace("labels", "frames"))
            file_ext = os.path.splitext(path)[1]
            assert file_ext in [".pkl", ".pickle"], "only pickle files are supported"
            with PathManager.open(path, "rb") as fopen:
                data = pickle.load(fopen)

            for vid_name in sorted(data.keys()):
                paths = [
                    os.path.join(root_dir, vid_name, str(item["Frame_id"]) + ".jpg")
                    for item in data[vid_name]
                ]
                im_ids = [item["unique_id"] for item in data[vid_name]]
                self.image_dataset.extend(paths)
                self.image_ids.extend(im_ids)

    def num_samples(self):
        """
        Size of the dataset
        """
        return self._num_samples

    def get_image_paths(self):
        """
        Get paths of all images in the datasets. See load_data()
        """
        self._load_data(self._path)
        if self.data_source == "disk_folder":
            assert isinstance(self.image_dataset, ImageFolder)
            return [sample[0] for sample in self.image_dataset.samples]
        else:
            return self.image_dataset

    @staticmethod
    def _replace_img_path_prefix(img_path: str, replace_prefix: str, new_prefix: str):
        if img_path.startswith(replace_prefix):
            return img_path.replace(replace_prefix, new_prefix)
        return img_path

    def __len__(self):
        """
        Size of the dataset
        """
        return self.num_samples()

    def __getitem__(self, idx):
        """
        - We do delayed loading of data to reduce the memory size due to pickling of
          dataset across dataloader workers.
        - Loads the data if not already loaded.
        - Sets and initializes the queue if not already initialized
        - Depending on the data source (folder or filelist), get the image.
          If using the QueueDataset and image is valid, save the image in queue if
          not full. Otherwise return a valid seen image from the queue if queue is
          not empty.
        """
        if not self.is_initialized:
            self._load_data(self._path)
            self.is_initialized = True
        if not self.queue_init and self.enable_queue_dataset:
            self._init_queues()
        is_success = True
        image_path = self.image_dataset[idx]
        image_id = self.image_ids[idx]
        try:
            if self.data_source == "disk_filelist" or "disk_filelist_surgery":
                image_path = self._replace_img_path_prefix(
                    image_path,
                    replace_prefix=self._remove_prefix,
                    new_prefix=self._new_prefix,
                )
                with PathManager.open(image_path, "rb") as fopen:
                    img = Image.open(fopen).convert("RGB")
            elif self.data_source == "disk_folder":
                img = self.image_dataset[idx][0]
            if is_success and self.enable_queue_dataset:
                self.on_sucess(img)
        except Exception as e:
            logging.warning(
                f"Couldn't load: {self.image_dataset[idx]}. Exception: \n{e}"
            )
            is_success = False
            # if we have queue dataset class enabled, we try to use it to get
            # the seen valid images
            if self.enable_queue_dataset:
                img, is_success = self.on_failure()
                if img is None:
                    img = get_mean_image(
                        self.cfg["DATA"][self.split].DEFAULT_GRAY_IMG_SIZE
                    )
            else:
                img = get_mean_image(self.cfg["DATA"][self.split].DEFAULT_GRAY_IMG_SIZE)
        
        return img, is_success, image_id
