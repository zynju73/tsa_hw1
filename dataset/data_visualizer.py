import numpy as np
import matplotlib.pyplot as plt

def data_visualize(dataset, t):
    """
    Choose t continous time points in data and visualize the chosen points. Note that some datasets have more than one
    channel.
    param:
        dataset: dataset to visualize
        t: the number of timestamps to visualize
    """

    data = dataset.data
    data = data[0]
    if data.ndim == 1:      #单通道
        data = np.expand_dims(data, axis=-1)

    T, C = data.shape
    if t > T:
        t = T
        print(f"Warning: t={t} is larger than data length T={T}.")
    plot_data = data[:t, :]
    channel_names = dataset.data_cols
    time_points = np.arange(t)



    if C == 1:        # 单通道：直接绘制
        plt.figure(figsize=(15, 6))
        plt.plot(time_points, plot_data[:, 0], label=channel_names[0])
        plt.title(f'{dataset.type} Dataset - {channel_names[0]} (First {t} Timestamps)')
        plt.ylabel(channel_names[0])

    elif C <= 4:# 少量通道：绘制在同一张图上
        plt.figure(figsize=(15, 6))
        for i in range(C):
            plt.plot(time_points, plot_data[:, i], label=channel_names[i], alpha=0.8)

        plt.title(f'{dataset.type} Dataset - All Channels (First {t} Timestamps)')
        plt.ylabel('Value')
        plt.legend(loc='upper right', ncol=2)

    else:
        num_subplots = 4
        fig, axes = plt.subplots(num_subplots, 1, figsize=(15, 12), sharex=True)
        fig.suptitle(f'{dataset.type} Dataset - Random Channel Subsets (First {t} Timestamps)', fontsize=16)

        all_channel_indices = list(range(C))
        # 选择前20 个通道进行分配
        channels_to_distribute = all_channel_indices[:min(16, C)]


        channels_per_subplot = len(channels_to_distribute) // num_subplots
        start_idx = 0
        for i in range(num_subplots):
            ax = axes[i]
            end_idx = start_idx + channels_per_subplot
            if i == num_subplots - 1:
                subplot_indices = channels_to_distribute[start_idx:]
            else:
                subplot_indices = channels_to_distribute[start_idx:end_idx]
            start_idx = end_idx

            for ch_idx in subplot_indices:
                ax.plot(time_points, plot_data[:, ch_idx],label=channel_names[ch_idx],alpha=0.4,linewidth=0.8)

            ax.set_title(f'Subset {i + 1} ({len(subplot_indices)} Channels)', fontsize=12)
            ax.set_ylabel('Value', rotation=0, labelpad=40, fontsize=12)
            ax.grid(True, alpha=0.5)
            ax.legend(loc='upper right', ncol=2, fontsize=8)

        plt.subplots_adjust(hspace=0.4)

    plt.xlabel('Time Step Index', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
