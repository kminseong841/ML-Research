# visualization.py
import numpy as np
import matplotlib.pyplot as plt

def plot_subplots(
    data_specs, 
    figsize=(15, 6), 
    suptitle=None, 
    tight=True,
    chunk_plots=False,   # <--- [NEW] 기본값=False
    chunk_size=5         # <--- [NEW] 기본값=5
):
    """
    Parameters
    ----------
    data_specs : list of dict
        각 subplot별로 그릴 정보를 담고 있는 딕셔너리들의 리스트.
        예: 
        data_specs = [
            {
                'plot_type': 'hist',  # 'hist' or 'line'
                'data_list': [array1, array2],
                'bins': 150,
                'alpha_list': [0.5, 0.5],
                'labels': ['Ground Truth','Prediction'],
                'xlabel': 'Ozone',
                'ylabel': 'Frequency',
                'title': 'Train O3 Distribution',
                'grid': False
            },
            {
                'plot_type': 'line',  # 'hist' or 'line'
                'x': x_values,
                'y_list': [mae_per_time, mse_per_time],
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
    chunk_plots : bool, optional
        True일 경우, data_specs를 chunk_size씩 끊어 여러 Figure로 나누어 그린다.
        (기본값: False)
    chunk_size : int, optional
        chunk_plots가 True일 때, 몇 개의 subplot을 한 Figure에 배치할지 결정한다.
        (기본값: 5)

    Returns
    -------
    None
    """

    # ------------------------------------
    # 1) chunk_plots=True 이고 data_specs 길이가 chunk_size 초과할 경우,
    #    여러 Figure로 나누어 그리는 로직
    # ------------------------------------
    if chunk_plots and len(data_specs) > chunk_size:
        # 예: 25개의 data_specs을 5개씩 끊으면 총 5개의 Figure가 나옴
        for start_idx in range(0, len(data_specs), chunk_size):
            chunk = data_specs[start_idx:start_idx + chunk_size]
            
            # 각 chunk(예: 최대 5개)에 대해 원래의 로직을 그대로 수행
            _plot_subplots_core(
                chunk,
                # chunk_size만큼 1행으로 그리므로, figsize는 여기서 동적으로 조정
                figsize=(chunk_size * 5, figsize[1]) if chunk_size > 1 else figsize,
                suptitle=suptitle,
                tight=tight
            )
        return  # 여러 Figure로 나누어 그린 후 종료

    # ------------------------------------
    # 2) 그 외(기존 방식 유지): 한 번에 한 Figure에 전부 그리기
    # ------------------------------------
    _plot_subplots_core(
        data_specs, 
        figsize=figsize, 
        suptitle=suptitle, 
        tight=tight
    )


def _plot_subplots_core(data_specs, figsize=(15, 6), suptitle=None, tight=True):
    """
    실제로 subplot을 그리는 내부 함수.
    (chunk로 나누지 않는 그리기 로직 그대로 모듈화)
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
            alpha_list = spec.get('alpha_list', [0.7] * len(data_list))
            labels = spec.get('labels', [None] * len(data_list))
            for data_idx, data_ in enumerate(data_list):
                ax.hist(
                    data_,
                    bins=bins,
                    alpha=alpha_list[data_idx],
                    label=labels[data_idx],
                    edgecolor='black'
                )

        elif plot_type == 'line':
            # 라인 플롯(시계열)
            x_values = spec.get('x', None)
            y_list = spec.get('y_list', [])
            labels = spec.get('labels', [None] * len(y_list))
            colors = spec.get('colors', [None] * len(y_list))
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