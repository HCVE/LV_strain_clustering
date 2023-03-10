from scipy import optimize
import tqdm
from itertools import groupby
from operator import itemgetter
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import *


# function that implements the piecewise linear interpolation to facilitate the separation of the strain curves
# with respect to the phases of the cardiac cycle. Based on the AIC we decide if the complexity of the model
# should be increased or not. Higher complexity means that the strain curves is split in more segments.
def segments_fit(X, Y, maxcount):
    xmin = X.min()
    xmax = X.max()
    n = len(X)
    aic_ = float('inf')
    r_ = None

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

        # Compute AIC/ BIC.
        aic = n * np.log10(err(r.x)) + 4 * count
        if aic < aic_:
            r_ = r
            aic_ = aic
        else:  # Stop.
            count = count - 1
            break
    # Return the last (n-1)
    return func(r_.x)


# check if the points to be used for linear regression are consecutive
def is_consecutive(points):
    if len(points) == 1:
        return False

    if all(np.diff(points) == 1):
        return True
    else:
        return False


def get_first_batch_consecutive(positions):
    for k, g in groupby(enumerate(positions), lambda d: d[0] - d[1]):
        temp = list(map(itemgetter(1), g))
        if len(temp) > 1:
            return temp
    return np.array([])


# identify and isolate the part of the strain curve that correspond to the systolic phase of the cardiac cycle
def isolate_systole(x_points, y_points):
    # find the position of the maximum values in the piecewise interpolation
    ind = np.argmax(y_points)
    # find the difference of the piecewise interpolation
    diff = np.diff(y_points[:ind + 1])
    # take only the positions with large positive differences as the ones corresponding to systolic phase
    pos = np.where(diff > 1)[0]
    pos = np.insert(pos, len(pos), pos[-1] + 1)

    if not is_consecutive(pos):
        pos = get_first_batch_consecutive(pos)

    systolic_points = x_points[pos]

    # delete the points that have already used
    x_points = np.delete(x_points, list(range(pos[-1])))
    y_points = np.delete(y_points, list(range(pos[-1])))
    return systolic_points, x_points, y_points


# identify and isolate the part of the strain curve that correspond to the diastolic phase of the cardiac cycle
def isolate_diastole(x_points, y_points):
    # find the difference of the piecewise interpolation
    diff_y = np.diff(y_points)
    diff_x = np.diff(x_points)
    slopes = diff_y / diff_x
    slopes = np.insert(slopes, 0, slopes[0])

    # take only the positions with large negative differences while above 80%
    # as the ones corresponding to diastolic phase
    pos = np.where((slopes < -25) & (x_points < 0.85))[0]

    if pos[0] != 0:
        pos = np.insert(pos, 0, pos[0] - 1)

    if not is_consecutive(pos):
        pos = get_first_batch_consecutive(pos)

    diastolic_points = x_points[pos]

    # delete the points that have already used
    x_points = np.delete(x_points, list(range(pos[-1])))
    y_points = np.delete(y_points, list(range(pos[-1])))
    return diastolic_points, x_points, y_points


# identify and isolate the part of the strain curve that correspond to the diastasis of the cardiac cycle
def isolate_diastasis(x_points, y_points):
    ind = np.where((x_points < 0.95))[0]

    if len(list(ind)) >= 2:
        diastasis_points = x_points[ind]
        # delete the points that have already used
        x_points = np.delete(x_points, list(range(ind[-1])))
        y_points = np.delete(y_points, list(range(ind[-1])))
    else:
        diastasis_points = np.array([])
    return diastasis_points, x_points, y_points


# identify and isolate the part of the strain curve that correspond to the late diastole of the cardiac cycle
def isolate_booster(x_points, y_points):
    # find the difference of the piecewise interpolation
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

    if list(pos):
        booster_points = x_points[pos]
        # delete the points that have already used
        x_points = np.delete(x_points, list(range(pos[1], pos[-1] + 1)))
        y_points = np.delete(y_points, list(range(pos[1], pos[-1] + 1)))
    else:
        booster_points = np.array([])
    return booster_points, x_points, y_points


# plot the individual strain curves with all phases annotated with specific colours.
# each phase is indicated between two vertical lines of the same colour.
# systole is represented with green, early diastole with blue, late diastole with black and diastasis with red.
def visualisation(time, data, pid, xpoints, ypoints, systolic_xpoints, diastolic_xpoints,
                  booster_xpoints, diastasis_xpoints, save_data_path):
    if not os.path.exists(os.path.join(save_data_path, "Splitting")):
        os.makedirs(os.path.join(save_data_path, "Splitting"))

    plt.figure()
    plt.plot(time, data, "-", label="Original Strain Curves")
    plt.plot(xpoints, -ypoints, "-or", label="Piecewise Linear Interpolation")

    plt.axvline(systolic_xpoints[0], color="green", alpha=0.6)
    plt.axvline(systolic_xpoints[-1], color="green", alpha=0.6)

    plt.axvline(diastolic_xpoints[0], color="blue", alpha=0.6)
    plt.axvline(diastolic_xpoints[-1], color="blue", alpha=0.6)

    if list(diastasis_xpoints):
        plt.axvline(diastasis_xpoints[0], color="red", alpha=0.6)
        plt.axvline(diastasis_xpoints[-1], color="red", alpha=0.6)

    plt.axvline(booster_xpoints[0], color="black", alpha=0.6)
    plt.axvline(booster_xpoints[-1], color="black", alpha=0.6)

    plt.xlabel("Time (% Cycle)")
    plt.ylabel("Strain (%)")
    plt.title("Fitted Linear Regression Model")
    plt.legend()
    plt.savefig(os.path.join(save_data_path, "Splitting", "Patient " + pid + ".png"))
    plt.savefig(os.path.join(save_data_path, "Splitting", "Patient " + pid + ".svg"))
    plt.close()


# fit a linear regression model to calculate the slopes during systole, early and late diastole.
def linear_regression_model(time, data, interval):
    start = np.argmin(np.abs(time - interval[0]))
    end = np.argmin(np.abs(time - interval[-1]))
    model = LinearRegression()
    model.fit(time[start:end].reshape(-1, 1), data[start:end].reshape(-1, 1))
    return model.coef_[0][0]


# extract the 6 features of the time series strain curves and store them in a pandas dataframe.
# By default, this function also stores the figures that indicate the parts of the strain curves
# that are corresponded to each phase of the cardiac cycle (set "do_plot=False" if the figures are not needed).
def extract_time_series_features(time, ts_data, names, path, do_plot=True):
    if not os.path.exists(path):
        os.makedirs(path)

    res = []
    for i in tqdm.tqdm(range(len(ts_data)), total=len(ts_data)):
        px, py = segments_fit(time[10:], np.abs(ts_data[i][10:]), 15)
        px = np.round(px, 2)
        py = np.round(py, 2)

        indexes = np.where(np.diff(px) <= 0)[0] + 1
        while list(indexes):
            px = np.delete(px, indexes)
            py = np.delete(py, indexes)
            indexes = np.where(np.diff(px) <= 0)[0] + 1

        systolic_xpoints, px1, py1 = isolate_systole(px, py)
        diastolic_xpoints, px1, py1 = isolate_diastole(px1, py1)
        booster_xpoints, px1, py1 = isolate_booster(px1, py1)
        diastasis_xpoints, px1, py1 = isolate_diastasis(px1, py1)

        systolic_slope = linear_regression_model(time, ts_data[i], systolic_xpoints)
        diastolic_slope = linear_regression_model(time, ts_data[i], diastolic_xpoints)
        booster_slope = linear_regression_model(time, ts_data[i], booster_xpoints)

        if list(diastasis_xpoints):
            diastasis_duration = diastasis_xpoints[-1] - diastasis_xpoints[0]
            diastasis_avg = (py[px == diastasis_xpoints[0]] + py[px == diastasis_xpoints[-1]])[0] / 2
        else:
            diastasis_duration = 0
            diastasis_avg = 0
        peak = max(np.abs(ts_data[i]))
        res.append([systolic_slope, diastolic_slope, booster_slope, diastasis_duration, diastasis_avg, peak])

        if do_plot:
            visualisation(time=time, data=ts_data[i], pid=names[i], xpoints=px, ypoints=py,
                          systolic_xpoints=systolic_xpoints, diastolic_xpoints=diastolic_xpoints,
                          booster_xpoints=booster_xpoints, diastasis_xpoints=diastasis_xpoints, save_data_path=path)
    return pd.DataFrame(data=res, columns=["Systolic Slope", "Diastolic Slope", "Booster Slope",
                                           "Diastasis Duration", "Diastasis Avg", "Peak"], index=names)
