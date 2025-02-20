import os
from scipy.io import loadmat

class MyIO:
    def __init__(self):
        pass

    def get_absolute_path(self, relative_path):
        environment_path = os.getcwd()
        full_path = os.path.join(environment_path, relative_path)
        print(full_path)
        return full_path

    def mat_reader(self, full_path):
        loaded_mat = loadmat(full_path)
        f_angles = loaded_mat['Angle']
        f_angular_velocity = loaded_mat['AngularVelocity']
        f_torques = loaded_mat['Torque']
        f_ultra_images = loaded_mat['Ultrasound']
        return f_angles, f_angular_velocity, f_torques, f_ultra_images


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    my_io = MyIO()
    path = my_io.get_absolute_path('Data/TS01_1/30deg_plant.mat')
    angles, velocity, torques , images = my_io.mat_reader(path)
    num = 10
    fig, ax = plt.subplots(1, num, figsize=(num * 7, num))
    for i in range(num):
        ax[i].imshow(images[:, :, i], cmap='gray')
        ax[i].axis('off')
        fig.tight_layout()
    plt.show()

