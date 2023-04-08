# Import the os module
import os
# Import the shutil module
import shutil
# Import the re module
import re

# Define a function named copy_img_folders that takes a folder path and a file name as parameters and copies all the img folders to the created md file folder
def copy_img_folders(folder_path, file_name):
  # Use the os.walk() function to iterate over all subdirectories in the folder and store the result in a variable
  walk_result = list(os.walk(folder_path))
  # Loop through each tuple in the list
  for dirpath, dirnames, filenames in walk_result:
    # Loop through each directory name in the current subdirectory
    for dir in dirnames:
      # Check if the directory name is "img"
      if dir == "img":
        # Get the source path of the img folder by joining it with the directory path
        src_path = os.path.join(dirpath, dir)
        # Get the destination path of the img folder by joining it with the file name and removing the extension
        dst_path = os.path.join(file_name[:-3], dir)
        # Copy the img folder to the destination path using shutil.copytree()
        shutil.copytree(src_path, dst_path)

# Define a function named merge_markdown_content that takes a folder path and a file name as parameters and merges all the markdown files content to the new file
def merge_markdown_content(folder_path, file_name):
  # Open a new markdown file in write mode using the with syntax
  with open(file_name, "w") as file3:
    # Use the os.walk() function to iterate over all subdirectories in the folder and store the result in a variable
    walk_result = list(os.walk(folder_path))
    # Sort the list of tuples by the first element (the directory path)
    walk_result.sort(key=lambda x: x[0])
    # Loop through each tuple in the sorted list
    for dirpath, dirnames, filenames in walk_result:
      # Sort the list of directory names in alphabetical order
      dirnames.sort()
      # Sort the list of file names in alphabetical order
      filenames.sort()
      # Loop through each file name in the current subdirectory
      for file in filenames:
        # Check if the file name ends with ".md"
        if file.endswith(".md"):
          # Open the markdown file in read mode using the with syntax
          with open(os.path.join(dirpath, file), "r") as file1:
            # Read the content of the markdown file
            content1 = file1.read()
            # Get the folder relative path by using the os.path.relpath() function
            section_name = os.path.relpath(dirpath, folder_path)
            # Write the section name as a header, followed by a newline character
            file3.write("# " + section_name + "\n")
            # Write the file name as the title of the content, followed by a newline character
            file3.write(file + "\n")
            # Replace any image path text in the content with the new location using re.sub()
            content1 = re.sub(r"img\/\w+\.(JPG|SVG|jpg|svg)", r"![\1](file3/\2)", content1)
            # Write the content to the new file
            file3.write(content1)

# Define a function named merge_markdown_files that takes a folder path and a file name as parameters and calls copy_img_folders and merge_markdown_content functions
def merge_markdown_files(folder_path, file_name):
  # Call copy_img_folders function with folder_path and file_name as arguments
  copy_img_folders(folder_path, file_name)
  # Call merge_markdown_content function with folder_path and file_name as arguments
  merge_markdown_content(folder_path, file_name)


# Call the function with the folder path and the file name as arguments
merge_markdown_files("../基础教程/A5-现代软件工程（更新中）", "现代软件工程.md")