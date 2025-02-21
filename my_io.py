import os
from scipy.io import loadmat, savemat

def get_absolute_path(relative_path):
    environment_path = os.getcwd()
    full_path = os.path.join(environment_path, relative_path)
    return full_path

def cut_images(images):
    height, width, _ = images.shape
    return images[5:(height - 19), 22:(width - 5), :]


def mat_skipper(some_mat):
    if "Ultrasound" not in some_mat:
        return True
    return False

def mat_saver(saver_path, loaded_mat, new_key, new_value):
    loaded_mat[new_key] = new_value
    del loaded_mat['Ultrasound']
    folders = saver_path.split('/')[1:]
    folders.pop()
    folder_path = '/'
    for folder in folders:
        folder_path = os.path.join(folder_path, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    savemat(saver_path, loaded_mat, do_compression=True)

def mat_reader(mode, file_path):
    if mode == 'absolute':
        loaded_mat = loadmat(file_path)
    else:
        full_path = get_absolute_path(file_path)
        loaded_mat = loadmat(full_path)
    if mat_skipper(loaded_mat):
        return None, None, None, None, None
    else:
        f_angles = loaded_mat['Angle']
        f_angular_velocity = loaded_mat['AngularVelocity']
        f_torques = loaded_mat['Torque']
        f_ultra_images = loaded_mat['Ultrasound']
        f_original = f_ultra_images
        f_ultra_images = cut_images(f_ultra_images)
        return f_angles, f_angular_velocity, f_torques, f_ultra_images, f_original

def data2folder(parent_path = 'Data'):
    parent_path = get_absolute_path(parent_path)
    child_folders1 = os.listdir(parent_path)
    child_folders2 = []
    for child_folder in child_folders1:
        if child_folder == '.DS_Store':
            continue
        else:
            child_folders2.append(os.path.join(parent_path, child_folder))
    return sorted(child_folders2)


def folder2mat(folder_path):
    f_mat_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".mat")])
    return f_mat_files

def find_factors(n):
    """找到 n 的所有因数"""
    factors = [i for i in range(1, n + 1) if n % i == 0]
    return factors

if __name__ == '__main__':
    # test area
    import matplotlib.pyplot as plt
    result = os.path.join(os.getcwd(), 'src', 'test_images')
    if not os.path.exists(result):
        os.makedirs(result)

    data_folders = data2folder()
    example_original, example_modified, example_txt = [], [], []
    start = 18
    while start < len(data_folders):
        example_original, example_modified, example_txt = [], [], []
        end = start+1
        counting = 0
        if end > len(data_folders):
            end = len(data_folders)
        print(end)
        for data_folder in data_folders[start:end]:
            mat_files = folder2mat(data_folder)
            counting += len(mat_files)
            for mat_file in mat_files:
                print(mat_file)
                angles, angular_velocity, torques, ultra_images, original = mat_reader(mode='absolute', file_path=mat_file)
                if original is not None:
                    print(ultra_images[:, :, 0].shape)
                    example_original.append(original[:, : ,0])
                    example_modified.append(ultra_images[:, :, 0])
                    example_txt.append(
                        f'{mat_file.split('/')[-2]} \n {(mat_file.split('/')[-1]).split('.')[0]}'
                    )
                else:
                    print('Warning! Skipping...', mat_file)
                    continue

        pointer = 0
        roll = 0
        avaliables = find_factors(counting)
        print(f'Overall {counting}', avaliables)
        dynamic_range = int(input("Choose range："))
        while pointer < len(example_original):
            fig, ax = plt.subplots(2, dynamic_range, figsize = (dynamic_range * 5, 11))
            for timer in range(dynamic_range):
                if pointer == len(example_original):
                    break
                ax[0, timer].imshow(example_original[pointer], cmap='gray')
                ax[0, timer].set_axis_off()
                ax[0, timer].set_title(example_txt[pointer], fontsize = 20, fontweight = 'bold')
                ax[1, timer].imshow(example_modified[pointer], cmap='gray')
                ax[1, timer].set_axis_off()
                pointer += 1
            fig.tight_layout()
            file_name2 = example_txt[pointer-1].split('\n')[0].replace(' ', '')
            plt.savefig(os.path.join(result, f'{file_name2}-{roll}.png'), format='png')
            plt.show()
            plt.close()
            roll += 1
        print(f'Done! {len(example_original)} examples | Read {counting} files')
        start += 1


