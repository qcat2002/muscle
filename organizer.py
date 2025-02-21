import os
from scipy.io import loadmat
from shutil import copytree

widths = [363, 448, 513, 598]
env_path = os.getcwd()
processed_folder = os.path.join(env_path, 'processed_data')
folder_363 = os.path.join(processed_folder, '363')
os.makedirs(folder_363, exist_ok=True)
folder_448 = os.path.join(processed_folder, '448')
os.makedirs(folder_448, exist_ok=True)
folder_513 = os.path.join(processed_folder, '513')
os.makedirs(folder_513, exist_ok=True)
folder_598 = os.path.join(processed_folder, '598')
os.makedirs(folder_598, exist_ok=True)

collector_363 = []
collector_448 = []
collector_513 = []
collector_598 = []
read_path = '/Users/zepeng/Project/muscle/Data1'
read_folders = [inner_name for inner_name in sorted(os.listdir(read_path)) if inner_name != '.DS_Store']
for read_folder in read_folders:
    mat_files = sorted([inner_file for inner_file in os.listdir(os.path.join(read_path, read_folder)) if inner_file.endswith(".mat")])
    print(f'正在读取 {os.path.join(read_path, read_folder)}, 共{len(mat_files)}个 mat 文件')
    first_mat = mat_files[0]
    temp_data = loadmat(os.path.join(read_path, read_folder, first_mat))
    images = temp_data['Cut_Ultrasound']
    height, width, channels = images.shape
    if width == 363:
        collector_363.append(read_folder)
    elif width == 448:
        collector_448.append(read_folder)
    elif width == 513:
        collector_513.append(read_folder)
    else:
        collector_598.append(read_folder)

# 363
print(len(collector_363))
print(len(collector_448))
print(len(collector_513))
print(len(collector_598))
for mat_folder in collector_363:
    original_path = os.path.join(read_path, mat_folder)
    target_path = os.path.join(folder_363, mat_folder)
    copytree(original_path, target_path)

for mat_folder in collector_448:
    original_path = os.path.join(read_path, mat_folder)
    target_path = os.path.join(folder_448, mat_folder)
    copytree(original_path, target_path)

for mat_folder in collector_513:
    original_path = os.path.join(read_path, mat_folder)
    target_path = os.path.join(folder_513, mat_folder)
    copytree(original_path, target_path)

for mat_folder in collector_598:
    original_path = os.path.join(read_path, mat_folder)
    target_path = os.path.join(folder_598, mat_folder)
    copytree(original_path, target_path)