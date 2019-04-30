import os
import sys
import hdfs

#hdfs.client._Request.webhdfs_prefix = "/webhdfs/api/v1"
_hdfs_client = None
_hdfs_url = "add your ip and port for pai hdfs here"
_hdfs_user_name = "add your user name here"
_hdfs_root = "/samples//"


def _hdfs_initialize():
    global _hdfs_client
    if _hdfs_client is None:
        _hdfs_client = hdfs.InsecureClient(_hdfs_url, root=_hdfs_root, user=_hdfs_user_name)
        _hdfs_client.set_permission(_hdfs_root, 777)

def _hdfs_download(hdfs_path, local_path):
    _hdfs_initialize()
    _hdfs_client.download(hdfs_path, local_path)

def _hdfs_upload(local_path, hdfs_path):
    _hdfs_initialize()
    _hdfs_client.makedirs(hdfs_path)
    _hdfs_client.upload(hdfs_path, local_path, overwrite=True)

def Download(hdfs_path, local_path):
    print("Downloading from {} to {}".format(hdfs_path, local_path))
    os.makedirs(local_path, exist_ok=True)
    _hdfs_download(hdfs_path, local_path)

def Upload(local_path, hdfs_path):
    print("Uploading from {} to {}".format(local_path, hdfs_path))
    _hdfs_upload(local_path, hdfs_path)