# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import os
import sys
import hdfs
import subprocess

hdfs_url = "YOUR_HDFS_URL" 
hdfs_user_name = "YOUR_HDFS_USERNAME"
root_dir = "YOUR_PROJECT_ROOT_DIR"

class HDFSHelper(object):

    def __init__(self, hdfs_url, hdfs_user_name, hdfs_root):
        self.__client = hdfs.InsecureClient(hdfs_url, root=hdfs_root, user=hdfs_user_name)
        self.__client.set_permission(hdfs_root,777)

    def Download(self, hdfs_path, local_path):
        print("Downloading from {} to {}".format(hdfs_path, local_path))
        os.makedirs(local_path, exist_ok=True)
        self.__client.download(hdfs_path, local_path)

    def Upload(self, local_path, hdfs_path):
        print("Uploading from {} to {}".format(local_path, hdfs_path))
        self.__client.makedirs(hdfs_path)
        self.__client.upload(hdfs_path, local_path, overwrite=True)

hdfsHelper = HDFSHelper(hdfs_url, hdfs_user_name, root_dir)

# Downloading data
hdfsHelper.Download(os.path.join(root_dir, "data"), ".")

# Call train.sh
subprocess.call("./train.sh", shell=True)

# Uploading data
jobName = os.environ['PAI_JOB_NAME']
output_dir = os.path.join(root_dir, "output", jobName)
hdfsHelper.Upload("./output/", output_dir)