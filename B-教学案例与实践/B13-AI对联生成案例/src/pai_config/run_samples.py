import os
import hdfshelper
import subprocess

jobName = os.environ['PAI_JOB_NAME'].split('~')[1]

print("Downloading data...")
hdfshelper.Download("/samples/data/", ".")
hdfshelper.Download("/samples/code/pip_requirements_gpu.txt", ".")

print("Running...")
subprocess.call("./train.sh", shell=True)

print("Uploading data...")
output_dir = os.path.join("/samples", "output", jobName)
hdfshelper.Upload("./output/", output_dir)
# hdfshelper.Upload("./data/checkpoint", output_dir)
# hdfshelper.Upload("./data/model.ckpt-200000.data-00000-of-00003", output_dir)
# hdfshelper.Upload("./data/model.ckpt-200000.data-00001-of-00003", output_dir)
# hdfshelper.Upload("./data/model.ckpt-200000.data-00002-of-00003", output_dir)
# hdfshelper.Upload("./data/model.ckpt-200000.index", output_dir)
# hdfshelper.Upload("./data/model.ckpt-200000.meta", output_dir)