import os
import shutil
import re

# Define a function named merge_markdown_files that takes a folder path and a file name as parameters and calls copy_img_folders and merge_markdown_content functions
def merge_markdown_files(folder_path, file_name,output_folder="output"):
  current_root_dir = os.path.dirname(os.getcwd())
  # Use the os.walk() function to iterate over all subdirectories in the folder and store the result in a variable
  walk_result = list(os.walk(folder_path))
  if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
  # Loop through each tuple in the list
  for dirpath, dirnames, filenames in walk_result:
    # Loop through each directory name in the current subdirectory
    for dir in dirnames:
      # Check if the directory name is "img"
      if dir == "img":
        # Get the source path of the img folder by joining it with the directory path
        src_path = os.path.join(dirpath, dir)
        relative_path = os.path.relpath( os.path.abspath(dirpath),current_root_dir)
        # print(f'relative_path {relative_path}')
        # Get the destination path of the img folder by joining it with the file name and removing the extension
        dst_path = os.path.join(output_folder,relative_path, dir)
        # Copy the img folder to the destination path using shutil.copytree()
        shutil.copytree(src_path, dst_path)
  # Use the os.walk() function to iterate over all subdirectories in the folder and store the result in a variable
  # Sort the list of tuples by the first element (the directory path)
  walk_result.sort(key=lambda x: x[0])
  # Open a new markdown file in write mode using the with syntax
  with open(os.path.join(output_folder, file_name), "w", encoding="utf-8") as output_file:
    # Loop through each tuple in the sorted list
    for dirpath, dirnames, filenames in walk_result:
      # Sort the list of directory names in alphabetical order
      dirnames.sort()
      # Sort the list of file names in alphabetical order
      filenames.sort()
      has_section_name = False
      # Loop through each file name in the current subdirectory
      for file in filenames:
        # Check if the file name ends with ".md"
        if file.endswith(".md"):
          if not has_section_name:
            # Get the folder relative path by using the os.path.relpath() function
            section_name = os.path.relpath(dirpath, folder_path)
            # Write the section name as a header, followed by a newline character
            output_file.write("# " + section_name.replace("\\","\n\n# ") + "\n\n")
            has_section_name = True
          # Open the markdown file in read mode using the with syntax
          with open(os.path.join(dirpath, file), "r", encoding="utf-8") as file1:
            # Read the content of the markdown file
            content = file1.read()
            # Write the file name as the title of the content, followed by a newline character
            output_file.write("## " +file.replace(".md","") + "\n\n")
            relative_path = os.path.relpath(os.path.abspath(dirpath),current_root_dir)
            # print(f'relative_path {relative_path}')
            # Replace any image path text in the content with the new location using re.sub()
            content = re.sub(r"img\/(\w+\.(JPG|SVG|jpg|svg|png))", f"{repr(relative_path)[1:-1]}/img/\g<1>", content)
            # Write the content to the new file
            output_file.write(content+ "\n")

# Call the function with the folder path and the file name as arguments
merge_markdown_files("../基础教程/A5-现代软件工程（更新中）", "现代软件工程.md")