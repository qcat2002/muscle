import matplotlib.pyplot as plt
import os

def plot_denoised_data_with_low_sample_rate(original_data, low_rate_data, indices, data_name=''):
    plt.figure(dpi=300)
    plt.title(f'Denoised-{data_name}-Sample Rate Reduction-{len(low_rate_data)} Samples')
    plt.plot(original_data, label='Original Data')
    plt.plot(indices, low_rate_data, label='Low Sample Rate Data')
    plt.legend()
    plt.savefig(os.path.join('src', 'readme_source', f'after denoising-{data_name.lower()}_low_sample_rate.png'))
    plt.show()