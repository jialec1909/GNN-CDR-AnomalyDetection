# Load the data from txt files as data structure.
import os
import re
import CDR

CDR_path = CDR.__path__[0]
data_path = f'{CDR_path}/../datasets'
visualization_folder = f"{CDR_path}/../datasets/merged_datasets/saved_images/observe"
os.makedirs(visualization_folder, exist_ok=True)
txt_files = [f for f in os.listdir(data_path) if f.endswith('.txt')]

import pandas as pd
import torch
import datetime
import matplotlib.pyplot as plt

def time_to_minutes(hour, minute):
    total_minutes = hour * 60 + minute
    return total_minutes
def save_plot_to_folder(plot, filename, output_folder):
    filepath = os.path.join(output_folder, filename)
    plot.savefig(filepath)
    plt.close()

def data_preprocess(path, visualization_folder):
    '''
    ---------------------------------------------
    :param path:
    :param visualization_folder:
    :return:
    ---------------------------------------------
    '''

    data = pd.read_csv(path,sep='\t',header=None)
    data = data.fillna(0)
    data.columns = ['cell_id', 'time_stp', 'country_code', 'SMS_in', 'SMS_out', 'Call_in', 'Call_out', 'Internet']
    data_nocc = data.drop(columns=['country_code'])

    #-------------------------------------------------------------------------------------------------------------------
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
    #-------------------------------------------------------------------------------------------------------------------



    # pick up zone == 4000 ---------------------------------------------------------------------------------------------
    zone_num_pick = 4000
    zone_data = data_nocc[data_nocc['cell_id'] == zone_num_pick]
    # process data of zone == 4000
    time_points = zone_data['time_in_minutes']
    sms_in = zone_data['SMS_in']
    sms_out = zone_data['SMS_out']
    call_in = zone_data['Call_in']
    call_out = zone_data['Call_out']
    internet_values = zone_data['Internet']
    # pick up zone == 4000 ---------------------------------------------------------------------------------------------

    # Visualisation: the data structure can be visualized by uncommenting the below codes.
    fig = plt.figure(figsize = (10, 6))
    plt.plot(time_points, sms_in, label = 'SMS_in')
    plt.plot(time_points, sms_out, label = 'SMS_out')
    plt.plot(time_points, call_in, label = 'Call_in')
    plt.plot(time_points, call_out, label = 'Call_out')
    plt.plot(time_points, internet_values, label = 'Internet')

    plt.xlabel('Time Points')
    plt.ylabel('Values')
    plt.title(f'Zone {zone_num_pick}: SMS, Call, and Internet Values over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    base_filename = os.path.basename(path)
    pattern = r'sms-call-internet-mi-(\d{4})-(\d{2})-(\d{2})'
    match = re.search(pattern, base_filename)
    image_name = match.group(0) if match else None
    filename = f"{image_name.replace('-', '_')}_visual_zone{zone_num_pick}_output.png"
    save_plot_to_folder(fig, filename, visualization_folder)

    return data_nocc


for txt_file in txt_files:
    file_path = os.path.join(data_path, txt_file)
    with open(file_path, 'r') as file:
        print(f"Open File: {txt_file}")
        data = data_preprocess(file_path, visualization_folder)
        print("----------------------------------------")
