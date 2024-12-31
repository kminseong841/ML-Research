# visualization.py
import numpy as np
import matplotlib.pyplot as plt

# 한 행에 여러 그래프(subplot)를 그리는 함수.
def plot_subplots(data_specs, figsize=(15, 6), suptitle=None, tight=True):
    """
    Parameters
    ----------
    data_specs : list of dict
        각 subplot별로 그릴 정보를 담고 있는 딕셔너리들의 리스트.
        예: 
        data_specs = [
            {   # 1번 subplot
                'plot_type': 'hist',   # 'hist' or 'line'
                'data_list': [ array1, array2 ],  # 여러 개도 가능 (예: 히스토그램 2개)
                'bins': 150,
                'alpha_list': [0.5, 0.5],
                'labels': ['Ground Truth','Prediction'],
                'xlabel': 'Ozone',
                'ylabel': 'Frequency',
                'title': 'Train O3 Distribution',
                'grid': False
            },
            {   # 2번 subplot
                'plot_type': 'line',   # 'hist' or 'line'
                'x': x_values,         # 시계열(혹은 x축)
                'y_list': [ mae_per_time, mse_per_time ],
                'labels': ['MAE','MSE'],
                'colors': ['blue','red'],
                'xlabel': 'Time Index',
                'ylabel': 'Error',
                'title': 'Time Series of Errors',
                'grid': True
            },
            ...
        ]
    figsize : tuple, optional
        (width, height) 형식으로 figure 크기.
    suptitle : str, optional
        전체 subplot에 대한 제목.
    tight : bool, optional
        True면 plt.tight_layout() 적용.

    Returns
    -------
    None
    """
    n_plots = len(data_specs)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    # subplot이 1개만 있을 경우 axes가 ndarray가 아니므로 처리
    if n_plots == 1:
        axes = [axes]

    for i, spec in enumerate(data_specs):
        ax = axes[i]

        plot_type = spec.get('plot_type', 'hist')
        title = spec.get('title', '')
        xlabel = spec.get('xlabel', '')
        ylabel = spec.get('ylabel', '')
        grid = spec.get('grid', False)
        
        if plot_type == 'hist':
            # 히스토그램 그리기
            data_list = spec.get('data_list', [])
            bins = spec.get('bins', 50)
            alpha_list = spec.get('alpha_list', [0.7]*len(data_list))
            labels = spec.get('labels', [None]*len(data_list))
            for data_idx, data_ in enumerate(data_list):
                ax.hist(data_, 
                        bins=bins, 
                        alpha=alpha_list[data_idx], 
                        label=labels[data_idx], 
                        edgecolor='black')
            
        elif plot_type == 'line':
            # 라인 플롯(시계열)
            x_values = spec.get('x', None)
            y_list = spec.get('y_list', [])
            labels = spec.get('labels', [None]*len(y_list))
            colors = spec.get('colors', [None]*len(y_list))
            for y_idx, y_data in enumerate(y_list):
                ax.plot(
                    x_values,
                    y_data,
                    label=labels[y_idx],
                    color=colors[y_idx] if colors[y_idx] else None
                )
        else:
            raise ValueError(f"지원하지 않는 plot_type: {plot_type}")

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(grid)

        # legend 처리
        # (만약 labels 중 하나라도 None이 아니면 legend를 표기하겠다고 가정)
        if spec.get('labels', None) is not None:
            ax.legend()

    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
    if tight:
        plt.tight_layout()
    plt.show()

# Ozone Map을 그리는 함수
def plot_ozone_maps(index, g_data, prediction, time_stamp, grid_size=32):
    ground_truth = g_data.reshape(-1, grid_size, grid_size)
    prediction = prediction.reshape(-1, grid_size, grid_size)
    error = ground_truth - prediction

    gt = ground_truth[index]
    pred = prediction[index]
    err = np.abs(error[index])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Ground Truth
    im1 = axes[0].imshow(gt, cmap='viridis', interpolation='nearest')
    axes[0].set_title('Ground Truth')
    plt.colorbar(im1, ax=axes[0])

    # Prediction
    im2 = axes[1].imshow(pred, cmap='viridis', interpolation='nearest')
    axes[1].set_title('Prediction')
    plt.colorbar(im2, ax=axes[1])

    # Error
    im3 = axes[2].imshow(err, cmap='viridis', interpolation='nearest')
    axes[2].set_title('Error')
    plt.colorbar(im3, ax=axes[2])

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    data_date = time_stamp[index].astype(str)
    fig.suptitle(data_date, fontsize=16)
    plt.tight_layout()
    plt.show()