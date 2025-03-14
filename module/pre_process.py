import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from scipy import stats
from math import sqrt

# =========================
# 1. 데이터 읽기 / 전처리 함수
# =========================
def read_processing(f1, f2):
    df1 = pd.read_csv(f1, usecols=lambda column: column != '지점', encoding='cp949')
    df2 = pd.read_csv(f2, usecols=['측정일시', '측정소명', '오존(ppm)'], encoding='cp949')
    df2 = df2[['측정소명', '측정일시', '오존(ppm)']]
    return df1, df2

def time_processing(df1, df2):
    df1['일시'] = pd.to_datetime(df1['일시'])
    df2['측정일시'] = pd.to_datetime(df2['측정일시'], format='%Y%m%d%H%M')
    df2.sort_values(by=['측정소명', '측정일시'], inplace=True)
    df2.reset_index(drop=True, inplace=True)

def interpol_processing(df, cols, region_df):
    specific_date = df.iloc[0]['일시']
    start_date = specific_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_date = (start_date + MonthEnd(0)).replace(hour=23, minute=0, second=0, microsecond=0)
    full_range = pd.date_range(start=start_date, end=end_date, freq='h')
    stations = df['지점명'].unique()

    # 결측된 부분 명확화
    new_index = pd.MultiIndex.from_product([stations, full_range], names=['지점명', '일시'])
    df.set_index(['지점명', '일시'], inplace=True)
    df = df.reindex(new_index).reset_index()

    # 방법1) 시점별 k개 이웃 관측소 기반 결측치 채우기
    # station -> (row, col) 매핑(dict) 만들기
    # region_df 는 met_regions_df 라고 가정
    station_coords = {}
    for i, row_data in region_df.iterrows():
        station_coords[row_data['region']] = (row_data['row'], row_data['col'])

    def get_distance(station1, station2):
        """두 지점(station)의 (row, col) 정보로부터 거리(유클리드) 계산"""
        r1, c1 = station_coords[station1]
        r2, c2 = station_coords[station2]
        return sqrt((r1 - r2)**2 + (c1 - c2)**2)

    k = 3  # 가까운 관측소 k개를 사용할지 설정(원하는 값으로 조정)

    # 일시별로 그룹화하여 각 시점에서 결측치가 있는 지점만 채우기
    df_grouped = df.groupby('일시', group_keys=False)
    filled_data = []

    for current_time, group in df_grouped:
        # group: 특정 시점(current_time)의 모든 지점 관측치
        # 결측을 발견하면, 같은 시점 내에서 가까운 k개 관측소 값을 평균 내어 사용
        for idx in group.index:
            station_name = group.loc[idx, '지점명']
            for col in cols:
                if pd.isnull(group.loc[idx, col]):
                    # -----------------------------
                    # (1) 결측 전 상태 출력 (주석 해제해서 사용)
                    print(f"[수정 후] <{col}> 결측 발견: {station_name} @ {current_time}")
                    # -----------------------------

                    # 1) 결측치가 아닌 관측소들 대상으로 거리 계산
                    valid_idx = group[~group[col].isna()].index
                    distances = []
                    for v_idx in valid_idx:
                        neighbor_station = group.loc[v_idx, '지점명']
                        dist_val = get_distance(station_name, neighbor_station)
                        distances.append((v_idx, dist_val))
                    # 2) 거리 순으로 정렬
                    distances.sort(key=lambda x: x[1])
                    # 3) 가까운 k개 관측소의 평균값을 결측치로 대체
                    nearest_idxs = [d[0] for d in distances[:k]]
                    fill_val = group.loc[nearest_idxs, col].mean()

                    # -----------------------------
                    # (2) 채울 값 확인 (주석 해제해서 사용)
                    print(f"→ 채우는 값: {fill_val} (가장 가까운 {k}개 관측소 평균)")
                    # -----------------------------    

                    group.loc[idx, col] = fill_val

                    # -----------------------------
                    # (3) 결측 치환 후 상태 출력 (주석 해제해서 사용)
                    updated_value = group.loc[idx, col]
                    print(f"[수정 후] <{col}> 치환 후 값: {updated_value} / 지점명: {station_name}, 시점: {current_time}")
                    print("-------------------------------------------------------------")
                    # -----------------------------
        filled_data.append(group)

    df = pd.concat(filled_data, axis=0) # 결측치가 없어진 group들을 연결해서 dataframe으로 재구성.

    # 방법2) 같은 지점 보간
    df.set_index('일시', inplace=True)
    df.sort_index(inplace=True)
    for station in df['지점명'].unique():
        station_df = df[df['지점명'] == station].sort_index()
        for col in cols:
            station_df[col] = station_df[col].interpolate(method='time')
            station_df[col] = station_df[col].ffill()
            station_df[col] = station_df[col].bfill()
        df.loc[df['지점명'] == station, cols] = station_df[cols]

    df.reset_index(inplace=True)
    df.sort_values(by=['지점명', '일시'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# =========================
# 2. 스케일링 / 변환 함수
# =========================
def scaling(df, columns_to_scal):
    scaler = MinMaxScaler()
    scaler.fit(df[columns_to_scal])
    df[columns_to_scal] = scaler.transform(df[columns_to_scal])
    return scaler

def log_scaling(df, columns_to_scal, epsilon_ls):
    i = 0
    for col in columns_to_scal:
        df[col] = np.log(df[col] + epsilon_ls[i])
        i += 1
    return epsilon_ls

def boxcox_scaling(df, columns_to_scal, scale_factor=1):
    lst = []
    for col in columns_to_scal:
        min_value = df[col].min()
        shift = 1 - min_value
        if min_value <= 0:
            df[col] = df[col] + shift
        df[col] = df[col] * scale_factor
        df[col], fitted_lambda = stats.boxcox(df[col])
        lst.append(fitted_lambda)
    lambda_lst = np.array(lst)
    return lambda_lst, shift

def standard_scaling(df, columns_to_scal):
    scaler = StandardScaler()
    scaler.fit(df[columns_to_scal])
    df[columns_to_scal] = scaler.transform(df[columns_to_scal])
    return scaler

def quantile_scaling(df, columns_to_scal):
    scaler = QuantileTransformer(output_distribution='normal', random_state=0)
    scaler.fit(df[columns_to_scal])
    df[columns_to_scal] = scaler.transform(df[columns_to_scal])
    return scaler


# =========================
# 3. 기타 유틸 함수 (좌표 변환, 매핑 등)
# =========================
def latlon_to_grid(lat, lon, lat_min, lat_max, lon_min, lon_max, grid_size=32):
    row = int((lat - lat_min) / (lat_max - lat_min) * (grid_size - 1))
    col = int((lon - lon_min) / (lon_max - lon_min) * (grid_size - 1))
    return row, col

def map_processing(df1, df2, grid_data, label_grid_data, time_stamp, f_cols,
                   met_regions_df, air_regions_df):
    time_size = 3
    n_day = len(df1['일시'].unique())
    n_f = len(f_cols)

    # df1 => 기상 데이터
    regions = met_regions_df['region'].unique()
    dates = df1['일시'].unique()

    # 기상 데이터 매핑
    for region in regions:
        grid_row = met_regions_df.loc[met_regions_df['region'] == region, 'row'].values[0]
        grid_col = met_regions_df.loc[met_regions_df['region'] == region, 'col'].values[0]
        met_sub_data = df1.loc[df1['지점명'] == region]

        # 시각별 데이터 할당
        # 0번째 시각
        grid_data[0, 0, grid_row, grid_col, 0:n_f] = met_sub_data.iloc[0][f_cols]
        grid_data[0, 1, grid_row, grid_col, 0:n_f] = met_sub_data.iloc[0][f_cols]
        grid_data[0, 2, grid_row, grid_col, 0:n_f] = met_sub_data.iloc[0][f_cols]
        # 1번째 시각
        grid_data[1, 0, grid_row, grid_col, 0:n_f] = met_sub_data.iloc[0][f_cols]
        grid_data[1, 1, grid_row, grid_col, 0:n_f] = met_sub_data.iloc[1][f_cols]
        grid_data[1, 2, grid_row, grid_col, 0:n_f] = met_sub_data.iloc[1][f_cols]
        # 2번째 시각
        grid_data[2, 0, grid_row, grid_col, 0:n_f] = met_sub_data.iloc[0][f_cols]
        grid_data[2, 1, grid_row, grid_col, 0:n_f] = met_sub_data.iloc[1][f_cols]
        grid_data[2, 2, grid_row, grid_col, 0:n_f] = met_sub_data.iloc[2][f_cols]

        #3번째~n_day-1까지 시각
        for date_index, date in enumerate(dates):
            if(date_index == n_day - time_size):
                break
            i_data = date_index+time_size # 3개의 시간대로 1개의 시간예측
            grid_data[i_data, 0:time_size, grid_row, grid_col, 0:n_f] = met_sub_data.iloc[date_index:date_index+time_size][f_cols]
    
    # df2 => 오염물질 농도 매핑
    label_regions = air_regions_df['region'].unique()
    label_dates = df2['측정일시'].unique()
    for region in label_regions:
        l_grid_row = air_regions_df.loc[air_regions_df['region'] == region, 'row'].values[0]
        l_grid_col = air_regions_df.loc[air_regions_df['region'] == region, 'col'].values[0]
        air_sub_data = df2.loc[df2['측정소명'] == region]

        # 시각별 데이터 할당
        for date_index, date in enumerate(label_dates):
            i_data = date_index
            label_grid_data[i_data, 0, l_grid_row, l_grid_col, 0] = air_sub_data.iloc[date_index]['오존(ppm)']
            time_stamp[i_data] = date # 현재 기록되는 오존값이 어느 시점의 오존 값인지 기록


    # 오존 feature 추가
    r_lst = air_regions_df['row'].tolist()
    c_lst = air_regions_df['col'].tolist()
    for i in range(len(r_lst)):
        r = r_lst[i]
        c = c_lst[i]

        # 데이터 할당
        # ex. grid_data: [0번째 데이터, 0번째시각, row=r, col = c, 1번째 feature]
        # ex label_grid_data: [0번째 데이터, 0번째 시각(단일시각), row=r, col=c, 0번쨰 feature(오존)]
        # grid_data는 각 데이터에 3개의 시각 저장, label_grid_data는 각 데이터의 1개의 시각 저장

        # 0번째 데이터
        grid_data[0, 0, r, c, n_f] = label_grid_data[0, 0, r, c, 0]
        grid_data[0, 1, r, c, n_f] = label_grid_data[0, 0, r, c, 0]
        grid_data[0, 2, r, c, n_f] = label_grid_data[0, 0, r, c, 0]
        # 1번째 데이터
        grid_data[1, 0, r, c, n_f] = label_grid_data[1, 0, r, c, 0]
        grid_data[1, 1, r, c, n_f] = label_grid_data[1, 0, r, c, 0]
        grid_data[1, 2, r, c, n_f] = label_grid_data[1, 0, r, c, 0]
        # 2번째 데이터
        grid_data[2, 0, r, c, n_f] = label_grid_data[0, 0, r, c, 0]
        grid_data[2, 1, r, c, n_f] = label_grid_data[1, 0, r, c, 0]
        grid_data[2, 2, r, c, n_f] = label_grid_data[2, 0, r, c, 0]

        #3번째~n_day-1 데이터
        for date_index in range(n_day - time_size):
            i_data = date_index+time_size # 3개의 시간대로 1개의 시간예측
            grid_data[i_data, 0:time_size, r, c, n_f] = label_grid_data[date_index:date_index+time_size, 0, r, c, 0]

# bit_mask 제작
def create_mask(df):
  mask = np.zeros((32,32), dtype=np.float32)
  r_lst = df['row'].tolist()
  c_lst = df['col'].tolist()
  for i in range(len(r_lst)):
    row = r_lst[i]
    col = c_lst[i]
    mask[row, col] = 1
  return mask

# 관측소가 있는 지점만 마스킹, flatten
def filtering_data(y, pred, mask):
    y_masked = y[:, mask!=0]
    pred_masked = pred[:, mask!=0]
    y_flat = y_masked.flatten()
    pred_flat = pred_masked.flatten()

    return y_flat, pred_flat
