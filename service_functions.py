# -*- coding: utf-8 -*-
import numpy as np
import collections
import os
import cv2

class Counter:
    """
        Counter class, just for beutifull code.
    """
    def __init__(self):
        self._data_ = -1


    def count(self):
        self._data_ = self._data_ + 1
        
        return self._data_


def get_box(frame, mask):
    """
        Returns sub-image by image and points of rectangle.
        
    :param frame: input image.
    :param mask: two points of rectungle.
    :return: sub image.
    """
    if mask is None:
        return frame.copy()
    pt1, pt2 = mask[0], mask[1]   
    return frame[np.int(pt1[1]):np.int(pt2[1]), np.int(pt1[0]):np.int(pt2[0])].copy()


def read_stat(name):
    """
    Service function, that reads a log file with points.

    :param name: path to the file.
    :return: 3 arrays with data.
    """
    file = open(name, 'r')
    I, X, Y = np.array([]), np.array([]), np.array([])
    for line in file:
        i, x, y = line.split('|')
        I = np.append(I, int(i))
        X = np.append(X, np.float32(x))
        Y = np.append(Y, np.float32(y))

    file.close()
    
    return X, Y, I


def stat(pts):
    """
    Returns the characteristics of the sample.

    :param pts: n-by-2 numpy array of n pts.
    :return: expected value and covariance matrix.
    """
    m = np.mean(pts, axis = 0)
    K = np.cov(pts[:,0], pts[:,1])
    
    return m, K


def get_mahalanobis_distance_sq_by_probability(alpha):
    """
    Returns a mahalanobis distance for current probability.

    :param alpha: probability.
    :return: mahalanobis distance.
    """
    return -2 * np.log(1 - alpha)


def get_mahalanobis_distance_sq(pt, m, K_inv):
    """
    Returns a mahalanobis distance for point pt and gaussian with.
    
    :param pt: point.
    :param m: expected value
    :param K_inv: invert of covariance matrix
    :return: mahalanobis distance.
    """
    v = pt - m
    return np.dot(np.dot(v, K_inv), v.transpose())


def indexes_of_points_that_mahalanobis_dist_smaller_than(pts, m, K, r_sq):
    """
    Return a array of boolean that have a true in position that point 
    (X[i], Y[i]) in r Mahalanobis distance.

    :param pts: n-by-2 numpy array of n pts.
    :param r: limit Mahalanobis distance.
    :return: boolean array.
    """
    
    L = pts.shape[0]
    indexes = np.zeros(L, dtype = np.int)
    K_inv = np.linalg.inv(K)
    for i in range(L):
        r_i_sq = get_mahalanobis_distance_sq(pts[i,:], m, K_inv)
        if r_i_sq < r_sq:
            indexes[i] = 1  
            
    return indexes == 1



def mean_filter(arr, w=3):
    """
        Realization of mean filter.
        
        :param arr: 1D-array of input data.
        :daram w: size of window in filter.
        :return: 1D-array that was filtered.
    """
    lenght = len(arr)
    arr_out = np.zeros(lenght, dtype = np.float)
    print(w)
    print(w//2)
    s = sum(arr[0:w])
    arr_out[w // 2] = s / w
    print('{} | {}'.format(1, s))
    for i in range(w // 2 + 1, lenght - w // 2):
        s -= arr[i - w//2]
        s += arr[i + w//2]
        arr_out[i] = s / w
        print('{} | {}'.format(i, s))
    return arr_out

def saveCalibrationOutput(m, K, mask, fname):
    file = open(fname, 'w')
    str1 = m[0].__str__() + '|' + m[1].__str__() + '\n'
    str2 = K[0][0].__str__() + '|' + K[0][1].__str__() + '|' + K[1][0].__str__() + '|' + K[1][1].__str__() + '\n'
    str3 = mask[0][0].__str__() + '|' + mask[0][1].__str__() + '|' + mask[1][0].__str__() + '|' + mask[1][1].__str__() + '\n'
    file.write(str1)
    file.write(str2)
    file.write(str3)
    file.close()
    
def readCalibrationOutput(fname):
    file = open(fname, 'r')
    data = collections.deque(maxlen=3)
    for line in file:
        data.append(line)
    file.close()
    m = np.zeros(2, dtype = np.float)
    m0, m1 = data[0].split('|')
    m[0] = np.float(m0)
    m[1] = np.float(m1)
    k = data[1].split('|')
    K = np.matrix(k[0] + ' ' + k[1] + ';' + k[2] + ' ' + k[3], dtype=np.float)
    a = data[2].split('|')
    arr0 = np.zeros(2, dtype=np.int)
    arr1 = np.zeros(2, dtype=np.int)
    arr0[0] = np.int(np.float(a[0]))
    arr0[1] = np.int(np.float(a[1]))
    arr1[0] = np.int(np.float(a[2]))
    arr1[1] = np.int(np.float(a[3]))
    return m, np.array(K), [arr0, arr1]

def mean_online(m_, data, i):
    return ((i + 1) * m_ + data) / (i + 2)

def line_parameters(pt1, pt2):
    x1 = pt1[0]
    y1 = pt1[1]
    x2 = pt2[0]
    y2 = pt2[1]
    k = (y1 - y2) / (x1 - x2)
    b = y1 - k * x1
    return k, b

def get_data_from_parsed_file(filename):
    ind = []
    lon = []
    lat = []
    dist = []
    v = []
    with open(filename, 'r') as file:
        t_prev = -1
        for line in file:
            p = line.split('|')
            tt = p[0].split(':')
            t_current = [int(tt[0]), int(tt[1]), int(tt[2])]
            lat_current = float(p[1])
            lon_current = float(p[2])
            dist_current = float(p[3])
            v_current = 0
            if t_prev != -1:
                t1 = 3600 * t_current[0] + 60 * t_current[1] + t_current[2]
                t2 = 3600 * t_prev[0] + 60 * t_prev[1] + t_prev[2]
                v_current = get_speed(dist_current, t1, t2)
            
            dist.append(dist_current)
            ind.append(t_current)
            lat.append(lat_current)
            lon.append(lon_current)
            v.append(v_current)     
                    
            t_prev = t_current
    return ind, lon, lat, dist, v

def get_speed(dist, time_1, time_2):
    v = float(3600 * dist / (time_1 - time_2))
    return v

def del_last_files():
    '''
        Service function for removing ald output of script.
        
    :return: None.
    '''
    dlist = os.listdir('output')
    for f in dlist:
        if f.split('.')[-1] == 'png':
            os.remove('output/' + f)
            
def is_in_side_of_rectangle(pt, pt1, pt2):
    '''
            To check point, is it inside rectangle,
        that top-left corner is pt1 and bottom-right is pt2.
    '''
    if pt1[0] < pt[0] < pt2[0] and pt1[1] < pt[1] < pt2[1]:
        return True
    return False

def modification_points(pts):
    '''
            To convert array of points to special array of points in 
        special notification in openCV.
    '''
    out = np.zeros((len(pts), 1, 2), dtype=np.float32)
    for i in range(len(pts)):
#        print(i)
        out[i][0][0] = pts[i][0]
        out[i][0][1] = pts[i][1]
    return out

def intensity_threshould_filter_of_points(img, points, threshould):
    points_out = collections.deque()
    points_out = []
    bool_points = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y, x, _ = img.shape

    for i in range(len(points)):
        for j in range(2, len(points[i])):
            p1 = points[i][j - 2]
            p2 = points[i][j]
            if is_in_side_of_rectangle(p2, (0, 0), (x, y)):
                I1 = int(gray[int(p1[1])][int(p1[0])])
                I2 = int(gray[int(p2[1])][int(p2[0])])
                if np.abs(I1 - I2) > threshould:
                    if [p1[0], p1[1]] not in bool_points:
                        points_out.append(p1)
                        bool_points.append([p1[0], p1[1]])
            else:
                break
    return points_out

def get_points(R, omega_0, omega_1, LINES, h, pt0 = [0, 0], x_min=-910, x_max=1080):#1920 - 910 = 1010
    omega = omega_0 + omega_1
    d_omega = omega / (LINES - 1)
#    N = R / 2
    dr = 2   
    count = 0
#    r = 0
    points = np.zeros(LINES, object)
    for i in range(LINES):
        points[i] = collections.deque()
    omega_i = 0
    for i in range(LINES):
        omega_i = np.pi / 2 - omega_0 + i * d_omega
        r = 0
        while r < R:
            pt = np.zeros(2, np.float32)
            pt[0] = int(r * np.cos(omega_i))
            pt[1] = int(r * np.sin(omega_i))
            if x_min <= pt[0] <= x_max:
                if pt[1] <= int(h[int(pt[0] - x_min)]):
                    points[i].append(pt + pt0)
            count += 1
            r += dr
    return points, count

