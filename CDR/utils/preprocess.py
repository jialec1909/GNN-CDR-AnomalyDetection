# Load the data from txt files as data structure.
import os
import re
data_path = '../datasets'
merged_data_path = '../datasets/merged_datasets/merged_txt'
os.makedirs(merged_data_path, exist_ok=True)
trans_merged_data_path = '../datasets/merged_datasets/merged_trans'
os.makedirs(trans_merged_data_path, exist_ok=True)
visualization_folder = "../datasets/merged_datasets/saved_images"
os.makedirs(visualization_folder, exist_ok=True)
txt_files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
merged_files = [f for f in os.listdir(merged_data_path) if f.endswith('.txt')]

import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--if_transformer', action='store_true', default = False,
                        help = 'If using transformer method to preprocess and sample data, False or True')
    parser.add_argument('--if_cover', action='store_true', default = False,
                        help = 'If cover the previous preprocessed data, False or True')

    opt = parser.parse_args()
    return opt


def time_to_minutes(hour, minute):
    total_minutes = hour * 60 + minute
    return total_minutes
def save_plot_to_folder(plot, filename, output_folder):
    filepath = os.path.join(output_folder, filename)
    plot.savefig(filepath)
    plt.close()

def data_preprocess(path, visualization_folder):
    data = pd.read_csv(path,sep='\t',header=None)
    data = data.fillna(0)
    data.columns = ['cell_id', 'time_stp', 'country_code', 'SMS_in', 'SMS_out', 'Call_in', 'Call_out', 'Internet']
    data_nocc = data.drop(columns=['country_code'])
    # Convert time stamps in milliseconds to seconds
    data_nocc['time_stp_seconds'] = data_nocc['time_stp'] / 1000
    # Create 'datetime' objects from the time stamps in seconds
    data_nocc['datetime_object'] = data_nocc['time_stp_seconds'].apply(datetime.datetime.fromtimestamp)
    # Format the 'datetime' objects as strings in the desired format
    data_nocc['formatted_time'] = data_nocc['datetime_object'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    data_nocc = data_nocc.groupby(['cell_id', 'formatted_time'], as_index=False).sum()
    data_nocc['dt_obj'] = pd.to_datetime(data_nocc['formatted_time'])
    # Extract the hour and minute from the 'datetime' objects and create new labels for them
    data_nocc['hour'] = data_nocc['dt_obj'].dt.hour
    data_nocc['minute'] = data_nocc['dt_obj'].dt.minute
    data_nocc["time_in_minutes"] = data_nocc.apply(lambda row: time_to_minutes(row["hour"], row["minute"]), axis=1)
    # Drop unnecessary columns (if needed)
    data_nocc.drop(['time_stp', 'time_stp_seconds','dt_obj'], axis=1, inplace=True)

    # Visualisation: the data structure can be visualized by uncommenting the below codes.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cell_ids = data_nocc['cell_id']
    time_stamps = data_nocc['formatted_time']
    time_stamps = pd.to_datetime(time_stamps)
    time_stamps = mdates.date2num(time_stamps)
    colors = data_nocc['SMS_in']  # 用SMS_in列的值作为颜色

    ax.scatter(cell_ids, time_stamps, colors, c=colors, cmap='viridis', marker='o')
    ax.set_xlabel('Cell ID')
    ax.set_ylabel('Time Stamp')
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    ax.set_zlabel('SMS_in')
    ax.set_title('3D Scatter Plot with SMS_in')
    base_filename = os.path.basename(path)
    pattern = r'sms-call-internet-mi-(\d{4})-(\d{2})-(\d{2})'
    match = re.search(pattern, base_filename)
    image_name = match.group(0) if match else None
    filename = f"{image_name.replace('call-internet-mi-', '').replace('-', '_')}_visual_output.png"
    save_plot_to_folder(plt, filename, visualization_folder)
    return data_nocc


def transformer_preprocessing(df):
    core_df = pd.DataFrame(df, columns = ["time_in_minutes",
                                          "cell_id",
                                          "SMS_in", "SMS_out", "Call_in", "Call_out",
                                          "Internet"])

    activity_means = core_df[['SMS_in', 'SMS_out', 'Call_in', 'Call_out', 'Internet']].mean()
    activity_std = core_df[['SMS_in', 'SMS_out', 'Call_in', 'Call_out', 'Internet']].std()

    core_df['SMS_in'] = (core_df['SMS_in'] - activity_means['SMS_in']) / activity_std['SMS_in']
    core_df['SMS_out'] = (core_df['SMS_out'] - activity_means['SMS_out']) / activity_std['SMS_out']
    core_df['Call_in'] = (core_df['Call_in'] - activity_means['Call_in']) / activity_std['Call_in']
    core_df['Call_out'] = (core_df['Call_out'] - activity_means['Call_out']) / activity_std['Call_out']
    core_df['Internet'] = (core_df['Internet'] - activity_means['Internet']) / activity_std['Internet']

    # core_df['merged_activity'] = (core_df['SMS_in'] / activity_means['SMS_in'] +
    #                               core_df['SMS_out'] / activity_means['SMS_out'] +
    #                               core_df['Call_in'] / activity_means['Call_in'] +
    #                               core_df['Call_out'] / activity_means['Call_out'] +
    #                               core_df['Internet'] / activity_means['Internet']) / 5
    return core_df

opt = parse_opt()
for txt_file in txt_files:
    file_path = os.path.join(data_path, txt_file)
    merged_path = os.path.join(merged_data_path, txt_file)
    if opt.if_cover:
        with open(file_path, 'r') as file:
            print(f"Open File: {txt_file}")
            data = data_preprocess(file_path, visualization_folder)
            data.to_csv(merged_path, index=False)
            print("-----------------Cover True & Previous Preprocess Covered-----------------------")
    else:
        pass

if opt.if_transformer:
    for merged_txt in merged_files:
        merged_dp = os.path.join(merged_data_path, merged_txt)
        trans_merged_dp = os.path.join(trans_merged_data_path, merged_txt)
        with open(merged_dp, 'r') as file:
            print(f"Open File: {merged_txt}")
            df = pd.read_csv(merged_dp)
            t_data = transformer_preprocessing(df)
            t_data.to_csv(trans_merged_dp, index = False)
    print("-----------------trans Ture & Preprocess Finished------------------")
