from urllib.request import urlretrieve
import time
import tarfile

URL_path = r"https://aiedugithub4a2.blob.core.windows.net/a2-data"
filename = r"Data.tar.gz"

print("Please input the local folder path:")
local = input()
from_path = URL_path + "/" + filename
to_path = local + "/" + filename

print("Downloading...")
try:
    urlretrieve(from_path, to_path)
    print("Done.")
    print("Extracting...")
    try:
        with tarfile.open(to_path) as file:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(file, path=local)
        print("Done.")
        print("All Work Finished!")
    except:
        print("Failed Extraction!")
except:
    print("Invalid Path!")
    
print("Exit in 3 seconds...")
time.sleep(3)
