import pandas as pd
import scipy
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from PIL import Image


#Temporal = True
#Spatial = False

folder_spatial_3D = 'W:\\Data\\data_processing_mi1354\\Frequency_Analysis\\Spatial\\3D\\'
folder_temporal_3D = 'W:\\Data\\data_processing_mi1354\\Frequency_Analysis\\Temporal\\3D\\'
folder_speed_3D = 'W:\\Data\\data_processing_mi1354\\Frequency_Analysis\\Speed\\3D\\'
folder_spatial_2D = 'W:\\Data\\data_processing_mi1354\\Frequency_Analysis\\Spatial\\2D\\'
folder_temporal_2D = 'W:\\Data\\data_processing_mi1354\\Frequency_Analysis\\Temporal\\2D\\'
folder_speed_2D = 'W:\\Data\\data_processing_mi1354\\Frequency_Analysis\\Speed\\2D\\'

def generateFmap(FlagTX, full_array):

    last_row = full_array.shape[0]
    last_col = full_array.shape[1]

    if FlagTX:
        axis = 1
        ftt_freq = 1/5000.0
        step_time = 5
        step_x = 1
        window_time = 2500
        window_x = 20
        nf = window_time
        thershold_spectrum = 8000


    else:
        axis = 0
        ftt_freq = 1.0/0.05
        step_time = 5
        step_x = 1
        window_time = 20
        window_x = 100
        nf = window_x
        thershold_spectrum = 8000


    len_time = len(np.arange(int(window_time / 2), last_row - int(window_time / 2), step_time))
    len_x = len(np.arange(int(window_x / 2), last_col - int(window_x / 2), step_x))
    freq_map = np.zeros((len_time, len_x))

    t = np.arange(0, len_time * (step_time / 5000.0), step_time / 5000.0)
    #t2 =  np.arange(0, len_x * (step_x / 5000.0), step_x / 5000.0)
    x = np.arange(0, len_x * (step_x *20), 20.0)

    index_x = 0
    for x_coord in np.arange((int(window_x / 2)), last_col - (int(window_x / 2)), step_x):
        index_y = 0
        for t_coord in np.arange((int(window_time / 2)), last_row - (int(window_time / 2)), step_time):
            roi = full_array[t_coord - (int(window_time / 2)):t_coord + (int(window_time / 2)),x_coord - (int(window_x / 2)):x_coord + (int(window_x / 2))]
            av_roi = np.average(roi, axis=axis)
            fourier = np.abs(fft(av_roi))
            fourier[0] = 0
            fourier = fourier[:int(len(fourier) / 2)]
            f = fftfreq(nf, ftt_freq)
            f = f[:int(len(fourier))]
            # if index_y < 10 and index_x == 0:
            #     plt.plot(f,fourier)
            #     plt.show()
            max = np.where(fourier == np.max(fourier))
            if np.max(fourier) < thershold_spectrum:
                main_freq = 0
            else:
                main_freq = f[max]

            freq_map[index_y, index_x] = main_freq
            print(index_x, index_y)
            index_y += 1
        index_x += 1
    return [freq_map,t,x]

def generateAVGraph(FlagFX, map):
    if FlagFX:
        axis = 1
        index = 1
    else:
        axis = 0
        index = 0

    median = np.median(map, axis = axis)

    for index_x in np.arange(0,map.shape[index]):
        index_t = 0
        if FlagFX:
            f = map[:, index_x]
        else:
            f = map[index_x, :]

        for ft in f:
            if FlagFX:
                diff = abs(ft - median[index_t])
                if median[index_t] == 0:
                    diff = 0
                else:
                    diff = diff/median[index_t]
                if diff > 0.9:
                    map[index_t, index_x] = median[index_t]

            else:
                diff = abs(ft - median[index_t])
                if median[index_t] == 0:
                    diff = 0
                else:
                    diff = diff/median[index_t]
                if diff > 0.9:
                    map[index_x, index_t] = median[index_t]

            index_t +=1

    av = np.average(map, axis = axis)
    return (av)

if __name__ == "__main__":
    for index in range(3,15,1):
        print(index)
        if index == 0:
            Exp_name = 'P42A_PT_Exp1'
            file_name_1 = 'W:\\Data\\data_processing_mi1354\\P42A_Parallel\\P42A_PT_Exp1\\Gabor1\\Cross_correlations.csv'
            file_name_2 = 'W:\\Data\\data_processing_mi1354\\P42A_Parallel\\P42A_PT_Exp1\\Corrected_Radiographs\\output-Corrected_Radiographs\\Cross_correlations.csv'

        if index == 1:
            Exp_name = 'P42A_PT_Exp2'
            file_name_1 = 'W:\\Data\\data_processing_mi1354\\P42A_Parallel\\P42A_PT_Exp2\\Corrected frames\\output-Corrected frames\\Cross_correlations.csv'
            file_name_2 = 'W:\\Data\\data_processing_mi1354\\P42A_Parallel\\P42A_PT_Exp2\\Gabor_last\\Cross_correlations.csv'

        if index == 2:
            Exp_name = 'P42A_PT_Exp3'
            file_name_1 = 'W:\\Data\\data_processing_mi1354\\P42A_Parallel\\P42A_PT_Exp3\\Gabor_1\\Cross_correlations.csv'
            file_name_2 = 'W:\\Data\\data_processing_mi1354\\P42A_Parallel\\P42A_PT_Exp3\\Corrected frames\\output-Corrected frames\\Cross_correlations.csv'

        if index == 3:
            Exp_name = 'P42A_PT_Exp4'
            file_name_1 = 'W:\\Data\\data_processing_mi1354\\P42A_Parallel\\P42A_PT_Exp4\\Gabor_1\\Cross_correlations.csv'
            file_name_2 = 'W:\\Data\\data_processing_mi1354\\Corrected_frames\\P42A_PT_Exp4\\output-Corrected_frames\\Cross_correlations.csv'

        if index == 4:
            Exp_name = 'P42A_ST_Exp1'
            file_name_1 = 'W:\\Data\\data_processing_mi1354\\P42A_Series\\P42A_ST_Exp1\\Gabor_1\\Cross_correlations.csv'
            file_name_2 = 'W:\\Data\\data_processing_mi1354\\Corrected_frames\\P42A_ST_Exp1\\output-Corrected_frames\\Cross_correlations.csv'

        if index == 5:
            Exp_name = 'P42A_ST_Exp2'
            file_name_1 = 'W:\\Data\\data_processing_mi1354\\P42A_Series\\P42A_ST_Exp2\\Gabor_1\\Cross_correlations.csv'
            file_name_2 = 'W:\\Data\\data_processing_mi1354\\Corrected_frames\\P42A_ST_Exp2\\output-Corrected_frames\\Cross_correlations.csv'

        if index == 6:
            Exp_name = 'P42A_ST_Exp3'
            file_name_1 = 'W:\\Data\\data_processing_mi1354\\P42A_Series\\P42A_ST_Exp3\\Gabor_1\\Cross_correlations.csv'
            file_name_2 = 'W:\\Data\\data_processing_mi1354\\Corrected_frames\\P42A_ST_Exp3\\output-Corrected_frames\\Cross_correlations.csv'

        if index == 7:
            Exp_name = 'P42A_ST_Exp4'
            file_name_1 = 'W:\\Data\\data_processing_mi1354\\P42A_Series\\P42A_ST_Exp4\\Gabor_1\\Cross_correlations.csv'
            file_name_2 = 'W:\\Data\\data_processing_mi1354\\Corrected_frames\\P42A_ST_Exp4\\output-Corrected_frames\\Cross_correlations.csv'

        if index == 8:
            Exp_name = 'M50_PT_Exp1'
            file_name_1 = 'W:\\Data\\data_processing_mi1354\\M50_Parallel\\M50_PT_Exp1\\Gabor_filtering\\Cross_correlations.csv'
            file_name_2 = 'W:\\Data\\data_processing_mi1354\\M50_Parallel\\M50_PT_Exp1\\Corrected_frames\\output-Corrected_frames\\Cross_correlations.csv'

        if index == 9:
            Exp_name = 'M50_PT_Exp2'
            file_name_1 = 'W:\\Data\\data_processing_mi1354\\M50_Parallel\\M50_PT_Exp2\\gabor_1\\Cross_correlations.csv'
            file_name_2 = 'W:\\Data\\data_processing_mi1354\\M50_Parallel\\M50_PT_Exp2\\Corrected_frames\\output-Corrected_frames\\Cross_correlations.csv'

        if index == 10:
            Exp_name = 'M50_PT_Exp3'
            file_name_1 = 'W:\\Data\\data_processing_mi1354\\M50_Parallel\\M50_PT_Exp3\\Corrected_frames\\output-Corrected_frames\\Cross_correlations.csv'
            file_name_2 = 'W:\\Data\\data_processing_mi1354\\M50_Parallel\\M50_PT_Exp3\\Gabor_2_1\\Cross_correlations.csv'

        if index == 11:
            Exp_name = 'M50_PT_Exp4'
            file_name_1 = 'W:\\Data\\data_processing_mi1354\\M50_Parallel\\M50_PT_Exp4\\corrected_frames\\output-corrected_frames\\Cross_correlations.csv'
            file_name_2 = 'W:\\Data\\data_processing_mi1354\\M50_Parallel\\M50_PT_Exp4\\corrected_frames\\output-corrected_frames\\Cross_correlations.csv'

        if index == 12:
            Exp_name = 'M50_ST_Exp1'
            file_name_1 = 'W:\\Data\\data_processing_mi1354\\M50_Series\\M50_ST_Exp1\\Gabor_1\\Cross_correlations.csv'
            file_name_2 = 'W:\\Data\\data_processing_mi1354\\M50_Series\\M50_ST_Exp1\\corrected_frames\\output-corrected_frames\\Cross_correlations.csv'

        if index == 13:
            Exp_name = 'M50_ST_Exp2'
            file_name_1 = 'W:\\Data\\data_processing_mi1354\\M50_Series\\M50_ST_Exp2\\Gabor_1\\Cross_correlations.csv'
            file_name_2 = 'W:\\Data\\data_processing_mi1354\\M50_Series\\M50_ST_Exp2\\corrected_frames\\output-corrected_frames\\Cross_correlations.csv'

        if index == 14:
            Exp_name = 'M50_ST_Exp3'
            file_name_1 = 'W:\\Data\\data_processing_mi1354\\M50_Series\\M50_ST_Exp3\\Gabor_1\\Cross_correlations.csv'
            file_name_2 = 'W:\\Data\\data_processing_mi1354\\M50_Series\\M50_ST_Exp3\\corrected_frames\\output-corrected_frames\\Cross_correlations.csv'

        if index == 15:
            Exp_name = 'M50_ST_Exp4'
            file_name_1 = 'W:\\Data\\data_processing_mi1354\\M50_Series\\M50_ST_Exp4\\Gabor_1\\Cross_correlations.csv'
            file_name_2 = 'W:\\Data\\data_processing_mi1354\\M50_Series\\M50_ST_Exp4\\corrected_frames\\output-corrected_frames\\Cross_correlations.csv'


        list_files = [file_name_1, file_name_2]
        list_arrays = []

        for file in list_files:
            print(file)
            list_col = []
            df = pd.read_csv(file)
            df_array = np.array(df)

            for column in df.columns:
                if 'Unnamed' in column:
                    list_col.append(column[8:])

            first_col = int(list_col[0]) + 2
            last_col = int(list_col[1])
            num_col = last_col - first_col
            x_cross = df_array[0:last_col, 0]
            df_array = df_array[:, first_col:last_col]
            list_arrays.append(df_array)

        print(np.max(list_arrays[0]), np.min(list_arrays[0]))
        print(np.max(list_arrays[1]), np.min(list_arrays[1]))

        rows_array1 = list_arrays[0].shape[0]
        rows_array2 = list_arrays[1].shape[0]

        if rows_array1 > rows_array2:
            list_arrays[0] = list_arrays[0][0:rows_array2,:]
        elif list_arrays[0].shape[0] < list_arrays[1].shape[0]:
            list_arrays[1] = list_arrays[1][0:rows_array1, :]


        full_array = np.concatenate((list_arrays[0], list_arrays[1]), axis=1)
        print(full_array.shape[1])


        if index == 11:
            full_array = list_arrays[0]
            print(full_array.shape[1])


        Temporal, t, x = generateFmap(True, full_array)
        plt.imshow(Temporal)
        save_folder = folder_temporal_3D + Exp_name
        np.savetxt(save_folder +'.txt', Temporal)
        im = Image.fromarray(Temporal)
        im.save(save_folder + '.tif')
        plt.savefig(save_folder)
        plt.close()
        graph = generateAVGraph(True, Temporal)
        save_folder = folder_temporal_2D + Exp_name
        save_data = np.array([t, graph])#, dtype=object)
        np.savetxt(save_folder +'.txt', save_data)
        plt.plot(t,graph)
        plt.title('Temporal domain')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.savefig(save_folder)
        plt.close()

        Spatial, t, x = generateFmap(False, full_array)
        plt.imshow(Spatial)
        save_folder = folder_spatial_3D + Exp_name
        np.savetxt(save_folder +'.txt', Spatial)
        im = Image.fromarray(Spatial)
        im.save(save_folder + '.tif')
        plt.savefig(save_folder)
        plt.close()

        graph = generateAVGraph(False, Spatial)
        save_folder = folder_spatial_2D + Exp_name
        save_data = np.array([x, graph])#, dtype=object)
        np.savetxt(save_folder +'.txt', save_data)
        plt.plot(x,graph)
        plt.title('Spatial domain')
        plt.xlabel('X-cross section [um]')
        plt.ylabel('Frequency [um-1]')
        plt.savefig(save_folder)
        plt.close()

        y_list = []
        x_list = []

        size_temporal_1 = Temporal.shape[0]
        print(size_temporal_1)
        y_list.append(size_temporal_1)
        size_spatial_1 = Spatial.shape[0]
        print(size_spatial_1)
        x_list.append(size_spatial_1)
        size_temporal_2 = Temporal.shape[1]
        y_list.append(size_temporal_2)
        print(size_temporal_2)
        size_spatial_2 = Spatial.shape[1]
        print(size_spatial_2)
        x_list.append(size_temporal_1)

        max_shape_y = np.max(y_list)
        max_shape_x = np.max(x_list)

        new_size = (max_shape_y, max_shape_x)
        print(new_size)

        im_temporal = Image.fromarray(Temporal)
        im_spatial = Image.fromarray(Spatial)


        print(im_spatial.getextrema())
        im_temporal = im_temporal.resize(new_size)
        im_spatial = im_spatial.resize(new_size)
        print(im_spatial.getextrema())

        temporal = np.array(im_temporal)
        spatial = np.array(im_spatial)

        spatial[spatial < 0] = 0
        temporal[temporal < 0] = 0
        spatial[spatial == 0] = 1



        speedMap = temporal/spatial
        speedMap[speedMap==temporal] = 0
        speedMap= np.abs(speedMap)
        save_folder = folder_speed_3D + Exp_name
        im = Image.fromarray(speedMap)
        im.save(save_folder + '.tif')
        plt.imshow(speedMap)
        plt.savefig(save_folder)
        plt.close()
        graph = generateAVGraph(True, speedMap)
        save_folder = folder_speed_2D + Exp_name
        t_new = np.arange(0,len(graph)*(5/5000.0),(5/5000.0))
        save_data = np.array(t_new)
        np.savetxt(save_folder + 'time.txt', save_data)
        save_folder = folder_speed_2D + Exp_name
        save_data = np.asarray(graph)
        np.savetxt(save_folder + 'graph.txt', save_data)
        plt.plot(t,graph)
        plt.title('Speed map')
        plt.xlabel('Time [s]')
        plt.ylabel('Speed [um/s]')
        plt.savefig(save_folder)
        plt.close()










