import os

def replace_img_width(folder_path):
    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".md"): continue
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            with open(file_path, 'w', encoding='utf-8') as f:
                for line in lines:
                    if '<img src="' in line and "/>" in line:
                        line = line.replace('/>', ' height=300/>')
                    f.write(line)
        elif os.path.isdir(file_path):
            replace_img_width(file_path)

folder = "D:\\GitHub\\ai-edu\基础教程\\A5-现代软件工程（更新中）\\第6步 - 设计与实现\\第14章 - 详细设计"
replace_img_width(folder)
