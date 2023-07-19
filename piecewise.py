from scipy import optimize
from itertools import groupby
from operator import itemgetter
from sklearn.linear_model import LinearRegression

import tqdm
import pandas as pd
import load_data as ld

from utils import *


def segments_fit(X, Y, maxcount):
    xmin = X.min()
    xmax = X.max()
    n = len(X)
    aic_ = float('inf')
    r_ = None

    # iteratively increase the number of segments in the piecewise linear model until the AIC does not decrease
    for count in range(7, maxcount + 1):
        seg = np.full(count - 1, (xmax - xmin) / count)
        px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.1].mean() for x in px_init])

        def func(p):
            seg = p[:count - 1]
            py = p[count - 1:]
            px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
            return px, py

        def err(p):  # This is RSS / n
            px, py = func(p)
            Y2 = np.interp(X, px, py)
            return np.mean((Y - Y2) ** 2)

        r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')

        # Compute AIC.
        aic = n * np.log10(err(r.x)) + 4 * count
        # if the AIC is lower than the previous adopt the higher complexity model.
        if aic < aic_:
            r_ = r
            aic_ = aic
        # if the new AIC isn't lower then stop and adopt the interpolated model calculated in the previous iteration.
        else:
            count = count - 1
            break

    return func(r_.x)


def is_consecutive(points):
    """Helper function to ascertain that the points used in linear regression method are consecutive"""
    if len(points) == 1:
        return False

    if all(np.diff(points) == 1):
        return True
    else:
        return False


def get_first_batch_consecutive(positions):
    """Helper function to return the first batch of consecutive points in an array. It is used when the points used
    in linear regression method are not consecutive"""
    for k, g in groupby(enumerate(positions), lambda d: d[0] - d[1]):
        temp = list(map(itemgetter(1), g))
        if len(temp) > 1:
            return temp
    return np.array([])


def isolate_systole(x_points, y_points):
    """Function to detect the points in the piecewise linear model that correspond to the systolic phase of the
    cardiac cycle """
    # find the position of the maximum values in the piecewise interpolation
    ind = np.argmax(y_points)
    # find the difference of the piecewise interpolation
    diff = np.diff(y_points[:ind + 1])
    # take only the positions with large positive differences as the ones corresponding to systolic phase
    pos = np.where(diff > 1)[0]
    pos = np.insert(pos, len(pos), pos[-1] + 1)

    # check if the selected positions are consecutive
    if not is_consecutive(pos):
        pos = get_first_batch_consecutive(pos)

    systolic_points = x_points[pos]

    # delete the points that have already used
    x_points = np.delete(x_points, list(range(pos[-1])))
    y_points = np.delete(y_points, list(range(pos[-1])))
    return systolic_points, x_points, y_points


def isolate_diastole(x_points, y_points):
    """Function to detect the points in the piecewise linear model that correspond to the early diastole of the
        cardiac cycle """
    # find the difference of the piecewise interpolation
    diff_y = np.diff(y_points)
    diff_x = np.diff(x_points)
    slopes = diff_y / diff_x
    slopes = np.insert(slopes, 0, slopes[0])

    # take only the positions with large negative differences while above 85%
    # as the ones corresponding to diastolic phase. The thresholds are selected empirically.
    pos = np.where((slopes < -25) & (x_points < 0.85))[0]

    if pos[0] != 0:
        pos = np.insert(pos, 0, pos[0] - 1)

    # check if the selected points are consecutive. If not take the first group of consecutive values to use them in
    # the linear interpolation model and calculate the desired slope.
    if not is_consecutive(pos):
        pos = get_first_batch_consecutive(pos)

    diastolic_points = x_points[pos]

    # delete the points that have already used
    x_points = np.delete(x_points, list(range(pos[-1])))
    y_points = np.delete(y_points, list(range(pos[-1])))
    return diastolic_points, x_points, y_points


def isolate_diastasis(x_points, y_points):
    # take the points of the piecewise linearly interpolated model that lie before the 95% of the cardiac cycle. This
    # threshold is used to ensure that the last point that definitely corresponds to late diastole. The previous
    # points have been deleted from the functions that detect systole and early diastole.
    ind = np.where((x_points < 0.95))[0]

    # There should be at least two points that occur before 95% of the cardiac cycle. Otherwise, diastasis is too
    # short to be detected and an empty array is returned.
    if len(list(ind)) >= 2:
        diastasis_points = x_points[ind]
        # delete the points that have already used
        x_points = np.delete(x_points, list(range(ind[-1])))
        y_points = np.delete(y_points, list(range(ind[-1])))
    else:
        diastasis_points = np.array([])
    return diastasis_points, x_points, y_points


def isolate_late_diastole(x_points, y_points):
    # calculate the difference of the piecewise interpolation points. It is used to find the point of large changes
    # in slope
    diff_y = np.diff(y_points)
    diff_x = np.diff(x_points)
    slopes = diff_y / diff_x
    slopes = np.insert(slopes, 0, slopes[0])

    pos = np.array([np.argmin(slopes)])
    if pos[0] == 0:
        pos = np.insert(pos, 1, pos[0] + 1)
    else:
        pos = np.insert(pos, 0, pos[0] - 1)

    if not is_consecutive(pos):
        if not np.any(get_first_batch_consecutive(pos)):
            pos = np.concatenate([np.where(slopes == max(slopes[pos]))[0] - 1, np.where(slopes == max(slopes[pos]))[0]],
                                 axis=0)
        else:
            pos = get_first_batch_consecutive(pos)

    # check if the list containing the break points from piecewise linear interpolation is empty.
    if list(pos):
        late_diastolic_points = x_points[pos]
        # delete the points that have already used
        x_points = np.delete(x_points, list(range(pos[1], pos[-1] + 1)))
        y_points = np.delete(y_points, list(range(pos[1], pos[-1] + 1)))
    else:
        late_diastolic_points = np.array([])
    return late_diastolic_points, x_points, y_points


def visualisation(time, data, pid, xpoints, ypoints, systolic_xpoints, diastolic_xpoints,
                  late_diastolic_xpoints, diastasis_xpoints, save_data_path):
    """Helper function to visualise the results from identification of each cardiac cycle. It is used for visual
    evaluation of the proposed feature extraction process. Its use is optional. """

    if not os.path.exists(os.path.join(save_data_path, "Splitting")):
        os.makedirs(os.path.join(save_data_path, "Splitting"))

    plt.figure()
    # plot the original strain curve, and its piecewise linearly interpolated approximation
    plt.plot(time, data, "-", label="Original Strain Curves")
    plt.plot(xpoints, -ypoints, "-or", label="Piecewise Linear Interpolation")

    # plot two vertical lines indicating the beginning and the ending of the systolic phase
    plt.axvline(systolic_xpoints[0], color="green", alpha=0.6)
    plt.axvline(systolic_xpoints[-1], color="green", alpha=0.6)

    # plot two vertical lines indicating the beginning and the ending of the early systolic phase
    plt.axvline(diastolic_xpoints[0], color="blue", alpha=0.6)
    plt.axvline(diastolic_xpoints[-1], color="blue", alpha=0.6)

    # diastasis can be too short and thus not detectable
    if list(diastasis_xpoints):
        # plot two vertical lines indicating the beginning and the ending of the diastasis
        plt.axvline(diastasis_xpoints[0], color="red", alpha=0.6)
        plt.axvline(diastasis_xpoints[-1], color="red", alpha=0.6)

    # plot two vertical lines indicating the beginning and the ending of the late diastolic phase
    plt.axvline(late_diastolic_xpoints[0], color="black", alpha=0.6)
    plt.axvline(late_diastolic_xpoints[-1], color="black", alpha=0.6)

    plt.xlabel("Time (% Cycle)")
    plt.ylabel("Strain (%)")
    plt.title("Cardiac Cycle Identification")
    plt.legend()
    plt.savefig(os.path.join(save_data_path, "Splitting", "Patient " + pid + ".png"))
    plt.savefig(os.path.join(save_data_path, "Splitting", "Patient " + pid + ".svg"))
    plt.close()


def linear_regression_model(time, data, interval):
    """ function that calculate a linear regression model. It is used to calculate the slope
    during systole, earle and late diastole"""
    # the starting point of the desired cardiac cycle phase
    start = np.argmin(np.abs(time - interval[0]))
    # the ending point of the desired cardiac cycle phase
    end = np.argmin(np.abs(time - interval[-1]))
    # fit the linear regression model
    model = LinearRegression()
    model.fit(time[start:end].reshape(-1, 1), data[start:end].reshape(-1, 1))
    return model.coef_[0][0]


def extract_time_series_features(time, ts_data, names, path, do_plot=False):
    """the main function that combines all the functions to extract the desired features from LV strain curves"""
    if not os.path.exists(path):
        os.makedirs(path)

    res = []
    # iterate in each time series recording
    for i in tqdm.tqdm(range(len(ts_data)), total=len(ts_data)):
        # extract the piecewise linear interpolation model
        px, py = segments_fit(time[10:], np.abs(ts_data[i][10:]), 15)
        px = np.round(px, 2)
        py = np.round(py, 2)

        # quality control to ensure that the piecewise linear interpolation doesn't have erroneous points that move
        # backwards in time
        indexes = np.where(np.diff(px) <= 0)[0] + 1
        while list(indexes):
            px = np.delete(px, indexes)
            py = np.delete(py, indexes)
            indexes = np.where(np.diff(px) <= 0)[0] + 1

        # extract the points that define each phase of the cardiac phase
        systolic_xpoints, px1, py1 = isolate_systole(px, py)
        diastolic_xpoints, px1, py1 = isolate_diastole(px1, py1)
        late_diastolic_xpoints, px1, py1 = isolate_late_diastole(px1, py1)
        diastasis_xpoints, px1, py1 = isolate_diastasis(px1, py1)

        # calculate the slopes in the desired phases
        systolic_slope = linear_regression_model(time, ts_data[i], systolic_xpoints)
        diastolic_slope = linear_regression_model(time, ts_data[i], diastolic_xpoints)
        late_diastolic_slope = linear_regression_model(time, ts_data[i], late_diastolic_xpoints)

        if list(diastasis_xpoints):
            diastasis_duration = diastasis_xpoints[-1] - diastasis_xpoints[0]
            diastasis_avg = (py[px == diastasis_xpoints[0]] + py[px == diastasis_xpoints[-1]])[0] / 2
        else:
            diastasis_duration = 0
            diastasis_avg = 0
        peak = max(np.abs(ts_data[i]))
        res.append([systolic_slope, diastolic_slope, late_diastolic_slope, diastasis_duration, diastasis_avg, peak])

        if do_plot:
            visualisation(time=time, data=ts_data[i], pid=names[i], xpoints=px, ypoints=py,
                          systolic_xpoints=systolic_xpoints, diastolic_xpoints=diastolic_xpoints,
                          late_diastolic_xpoints=late_diastolic_xpoints, diastasis_xpoints=diastasis_xpoints, save_data_path=path)
    return pd.DataFrame(data=res, columns=["Systolic Slope", "Diastolic Slope", "Booster Slope",
                                           "Diastasis Duration", "Diastasis Avg", "Peak"], index=names)


def retrieve_strain(root_path, full_path, avc, p_waves):
    # read the time, strain and ECG data from the .txt files
    original_data, data, patient_id, interval = ld.read_data(full_path)

    # Obtain the time AVC from the respective .xlsx file, as annotated manually be an expert
    # IDs of patients that do not have a measurement are included in the "excluded_patients1" variable
    excluded_patients1, avc_times = ld.read_avc_time(root_path, avc)

    # Read the time at which the peak of the P-wave occurs, as annotated manually be an expert
    # IDs of patients that do not have a measurement are included in the "excluded_patients2" variable
    excluded_patients2, p_wave_times = ld.read_p_wave_data(root_path, p_waves)

    # Remove from the time, strain and ECG data, the measurements that correspond to IDS that should be excluded
    original_data, data, patient_id, interval = exclude_patients(excluded_patients1, excluded_patients2, original_data,
                                                                 data, patient_id, interval)
    return original_data, patient_id, interval, avc_times, p_wave_times
