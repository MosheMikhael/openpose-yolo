import cv2
import numpy as np
import service_functions as sf

# variables for color coding.
red = (0, 0, 255)
dark_red = (0, 0, 204)
green = (0, 255, 0)
dark_green = (0, 100, 0)
blue = (255, 0, 0)
cyan = (255, 255, 0)

purple = (255, 0, 255)
orange = (0, 165, 255)
yellow = (0, 255, 255)
white = (255, 255, 255)
black = (0, 0, 0)
gold = (0, 215, 255)
gray = (220,220,220)

def draw_point(img, pt, radius=5, color=red, thickness=1):
    """
    Drawing point on image.

    :param img: source image, array of coordinates of points.
    :param pt: array of coordinates of point.
    :param radius: radius of point.
    :param color: color by opencv signature (not (r, g, b), but (b, g, r)).
    :param thickness: thickness of line.
    :return: image as numpy array of points with drawn point.
    """
    out = img#.copy()
    out = cv2.circle(out, (int(pt[0]), int(pt[1])), radius=radius, 
                                         color=color, thickness=thickness)
                                     
    return out


def draw_points(img, pts, radius=3, color=red, thickness=1):
    """
        Drawing points on image.

    :param img: source image, array of coordinates of points.
    :param pts: array of arrays of coordinates of point.
    :param radius: radius of point.
    :param color: color by opencv signature (not (r, g, b), but (b, g, r)).
    :param thickness: thickness of line.
    :return: image as numpy array of points with drawn points.
    """
    out = img#.copy()
    for pt in pts:
        out = draw_point(out, pt, radius, color, thickness)
        
    return out


def draw_arrow(img, pt1, pt2, color=green, thickness=1):
    """
        Drawing arrow on image.

    :param img: source image, array of coordinates of points.
    :param pt1: coordinates of first point.
    :param pt2: coordinates of first second.
    :param color: color by opencv signature (not (r, g, b), but (b, g, r)).
    :param thickness: thickness of line.
    :return: image as numpy array of points with drawn arrow.
    """
    out = img#.copy()
    out = cv2.arrowedLine(out, (int(pt1[0]), int(pt1[1])), 
                                (int(pt2[0]), int(pt2[1])), color, thickness)
    return out


def draw_arrows(img, pts1, pts2, color=green, thickness=1):
    """
        Drawing arrows on image.

    :param img: source image, array of coordinates of points.
    :param pts1: array of first points coordinates.
    :param pts2: array of second points coordinates.
    :param color: color by opencv signature (not (r, g, b), but (b, g, r)).
    :param thickness: thickness of lines.
    :return: image as numpy array of points with drawn arrows.
    """
    out = img
    for i in range(len(pts1)):
        out = draw_arrows(out, pts1[i], pts2[i], color, thickness)
        
    return out


def draw_line(img, k, b, color=green, thickness=1):
    """
        Draws a line using equation. (y = k*x + b)

    :param img: source image, array of coordinates of points.
    :param k: first coefficient in equation.
    :param b: second coefficient in equation.
    :param color: color by opencv signature (not (r, g, b), but (b, g, r)).
    :param thickness: thickness of line.
    :return: image as numpy array of points with drawn line.
    """
    out = img#.copy()
#    print(type(b), b)
    if np.isnan(b) or np.isnan(k) or np.isinf(b):
        return out
    x0 = 0
    y0 = int(b)
    x1 = out.shape[1]
    y1 = int(k*x1 + b)
    out = cv2.line(out, (x0, y0), (x1, y1), color, thickness)
    return out


def draw_lines(img, k, b, color=green, thickness=1):
    """
        Draws lines on images, using equation: y = kx + b

    :param img: source image, array of coordinates of points.
    :param k: array of first coefficients in equation.
    :param b: array of second coefficients in equation.
    :param color: color by opencv signature (not (r, g, b), but (b, g, r)).
    :param thickness: thickness of lines.
    :return: image as numpy array of points with drawn line.
    """
    out = img#.copy()
    for i in range(len(b)):
        if b[i]:
            out = draw_line(out, k[i], b[i], color, thickness)
            
    return out


def draw_ellipse(img, pt, axis_x, axis_y, color=blue, thickness=1):
    """
        Draws a ellipse on image. (Draws a simple or thick elliptic arc or 
        fills an ellipse sector.)

    :param img: source image, array of coordinates of points.
    :param pt: center of ellipse, coordinates.
    :param axis_x: half of the size of the ellipse main axis. (axis x)
    :param axis_y: half of the size of the ellipse main axis. (axis y)
    :param color: color by opencv signature (not (r, g, b), but (b, g, r)).
    :param thickness: thickness of arc.
    :return: image as numpy array of points with drawn ellipse.
    """
    out = img#.copy()
    out = cv2.ellipse(out, (int(pt[0]), int(pt[1])), 
                              (int(axis_x), int(axis_y)), 
                                        0, 0, 360, color, thickness)
                                        
    return out


def draw_rectangle(img, mask, color=red, thickness=1):
    """
        Draws a rectangle on image.

    :param img: source image, array of coordinates of points.
    :param mask :a 2-list the form [top_left, bottom_right], where top_left
     is the top_left vertex of the desired mask in the image, and 
     bottom_right is its bottom right vertex (the mask may be all the image).
    :param color: color by opencv signature (not (r, g, b), but (b, g, r)).
    :param thickness: thickness of lines.
    :return: image as numpy array of points with drawn rectangle.
    """
    out = img#.copy()
    [pt1, pt2] = mask
    out = cv2.rectangle(out, (int(pt1[0]), int(pt1[1])), 
                                (int(pt2[0]), int(pt2[1])), 
                                    color=color, thickness=thickness)

    return out


def draw_text(img, pt, text, color=black, font_scale=0.5, line_type=2):
    """
        Draws text.

    :param img: source image, array of coordinates of points.
    :param pt: top-left corner.
    :param text: text.
    :param color: color by opencv signature (not (r, g, b), but (b, g, r)).
    :param font_scale: font scale.
    :param line_type: line thickness.
    :return: image as numpy array of points with drawn text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner_of_text = (int(pt[0]), int(pt[1]))
    out = img#.copy()
    out = cv2.putText(out, text, bottom_left_corner_of_text, font, 
                                          font_scale, color, line_type)
                                              
    return out


def draw_axis(img, pt0, sigma_x, sigma_y, rec_pt_1=None, rec_pt_2=None):
    """
        Draws axit and another things.

    :param img: source image, array of coordinates of points.
    :param pt0: center of coordinates.
    :param sigma_x: half of the size of the ellipse main axes. (axis x)
    :param sigma_y: half of the size of the ellipse main axes. (axis y)
    :param rec_pt_1: top-left corner of rectangle.
    :param rec_pt_2: bottom-right corner of rectangle.
    :return: image as numpy array of points with drawn axis and another information.
    """
    # center of axis
    out = draw_point(img, pt=pt0, color=dark_green, thickness=2)
    # axis x
    out = cv2.line(out, (0, int(pt0[1])), (out.shape[1], int(pt0[1])), 
                                                               color=white)
    # axis y
    out = cv2.line(out, (int(pt0[0]), 0), (int(pt0[0]), out.shape[0]), 
                                                               color=white)
    # 1 sigma
    out = draw_ellipse(out, pt0, sigma_x, sigma_y, color=yellow)
    # 3 sigma
    out = draw_ellipse(out, pt0, 3*sigma_x, 3*sigma_y, color=red)
    # rectangle of searching area
    if rec_pt_1 is not None:
        out = draw_rectangle(out, rec_pt_1, rec_pt_2, color=cyan)
    return out


def draw_axis_by_sample(img, X, Y, rec_pt_1=None, rec_pt_2=None):
    """
        Draw axes by sample.

    :param img: input image.
    :param X: x-coordinates sample.
    :param Y: y-coordinates sample.
    :param rec_pt_1: top-left corner of rectangle.
    :param rec_pt_2: bottom-right corner of rectangle.
    :return: output image.
    """
    m, K = sf.stat(X, Y)
    return draw_axis(img, m, np.sqrt(K[0][0]), np.sqrt(K[1][1]), 
                                                     rec_pt_1, rec_pt_2)

#OBSOLETE
def draw_all(img, pt, pt0, sigma_x, sigma_y, text = None, 
                                         rec_pt1 = None, rec_pt2 = None):
    """

    :param img: input image.
    :param pt: current point.
    :param pt0: middle point.
    :param sigma_x: x-sigma of sample.
    :param sigma_y: y-sigma of sample.
    :param text: print text.
    :param rec_pt1: top-left corner of rectangle.
    :param rec_pt2: bottom-right corner of rectangle.
    :return: output image.
    """
    out = draw_axis(img, pt0, sigma_x, sigma_y, rec_pt1, rec_pt2)
    out = draw_arrow(out, pt0, pt)
    out = draw_point(out, pt, radius=3, thickness=6, color=red)
    out = draw_point(out, (pt[0], pt0[1]), radius=2, thickness=2, color=white)
    out = draw_point(out, (pt0[0], pt[1]), radius=2, thickness=2, color=white)
    if text is not None:
        out = draw_text(out, (0, 50), **text)
        text.pop('text')
        text.pop('color')
        out = draw_text(out, (0, 80), 
                        text='x = ' + str(pt[0] - pt0[0]), color=black,
                                                                 **text)
        out = draw_text(out, (0, 110), 
                        text='y = ' + str(-pt[1] + pt0[1]), color=black,
                                                                    **text)
    return out


def draw_mahalanobis_ellipse(img, r, m, K, color=cyan, thickness=1, 
               draw_center_pt = True, draw_extremum_pts = True, 
               draw_axes = True):
    """
        Drawing ellipse with r mahalanubis radius.

    :param img: input image.
    :param r: radius.
    :param m: center of ellipse.
    :param K: correlation matrix.
    :param color: color of points.
    :param thickness: thickness of points.
    :return: output image.
    """
    out = img#.copy()
    
    '''
    sigma_x = np.sqrt(K[0][0])
    sigma_y = np.sqrt(K[1][1])
    mu_x = m[0]
    mu_y = m[1]
    rho = K[0][1] / (sigma_x * sigma_y)
    x_min, x_max, y_min, y_max = 99999, m[0], 99999, m[1]     #?????????????? inversion ?
    d_max_sq = (x_max - m[0]) ** 2 + (y_max - m[1]) ** 2
    d_min_sq = (x_min - m[0]) ** 2 + (y_min - m[1]) ** 2
    for theta in np.arange(0, 360):
        x = r * sigma_x * np.cos(theta) + mu_x
        y = r * sigma_y * (rho * np.cos(theta) + np.sqrt(1 - rho ** 2) * np.sin(theta)) + mu_y
        d_sq = (x - m[0]) ** 2 + (y - m[1]) ** 2
        if d_sq < d_min_sq:
            d_min_sq = d_sq
            x_min = x
            y_min = y
        if d_sq > d_max_sq:
            d_max_sq = d_sq
            x_max = x
            y_max = y
        out = draw_point(out, (x, y), color=color, thickness=thickness, radius=2)
    
    out = draw_point(out, (x_min, y_min), color=blue, thickness=thickness+1, radius=2)
    out = draw_point(out, (x_max, y_max), color=red, thickness=thickness+1, radius=2)
    '''
    
    M = np.linalg.cholesky(K) #cholesky decomposition of K
    theta = [i*np.pi/180 for i in range(0, 360)]
    param = np.concatenate(([np.cos(theta)], [np.sin(theta)]), axis = 0)
    param_points = r * M.dot(param) + np.array([ [m[0]], [m[1]] ])
    param_points = np.array(param_points, dtype = np.int)
    polygone = (param_points.T).reshape((-1, 1, 2))
    out = cv2.polylines(out, [polygone], True, color, thickness)   
    m_int = np.array(m, dtype = np.int)
                       
    #drawing center
    if draw_center_pt:
        out = draw_point(out, m_int, color=red, 
                                 thickness=thickness+1, radius=2)
                                 
    #drawing extremum points:                                 
    eigVal, eigVect = np.linalg.eig(K)
    scaledEigVect = r * eigVect * np.sqrt(eigVal)   
    scaledEigVect = np.array(scaledEigVect, dtype = np.int)
    v1, v2 = scaledEigVect[:,0], scaledEigVect[:,1]    
    M1 = tuple((m_int + v1).tolist())
    Mm1 = tuple((m_int - v1).tolist())

    M2 = tuple((m_int + v2).tolist())
    Mm2 = tuple((m_int - v2).tolist())

    if eigVal[0] > eigVal[1]:
        M1, M2 = M2, M1    
        Mm1, Mm2 = Mm2, Mm1
            
    if draw_extremum_pts:          
        out = draw_points(out, [M1, Mm1, M2, Mm2], color = yellow, 
                               thickness=thickness+1, radius=2)                                
    #drawing axes:
    if draw_axes:
        out = cv2.line(out, M1, Mm1, blue, thickness = thickness)
        out = cv2.line(out, M2, Mm2, blue, thickness = thickness)

        
    return out

def draw_grid(img, grid, color=cyan, thickness=1):
    '''
        Draw grid on image by special variable 'grid'.
    '''
    rows = len(grid)
    cols = len(grid[0])
    out = img#.copy()
#    out = draw_rectangle()
    for i in range(rows):
#        for j in range(cols):
        pt1 = (grid[i][0][0][0], grid[i][0][0][1])
        pt2 = (grid[i][cols - 1][1][0], grid[i][cols - 1][0][1])
        out = cv2.line(out, pt1, pt2, color=color, thickness = thickness)
    for j in range(cols):
        pt1 = (grid[0][j][0][0], grid[0][j][0][1])
        pt2 = (grid[rows - 1][j][0][0], grid[rows - 1][j][1][1])
        out = cv2.line(out, pt1, pt2, color=color, thickness = thickness)
    out = cv2.line(out, (grid[0][cols - 1][1][0], grid[0][cols - 1][0][1]), (grid[rows - 1][cols - 1][1][0], grid[rows - 1][cols - 1][1][1]), color=color, thickness = thickness)
    out = cv2.line(out, (grid[rows - 1][0][0][0], grid[rows - 1][0][1][1]), (grid[rows - 1][cols - 1][1][0], grid[rows - 1][cols - 1][1][1]), color=color, thickness = thickness)
    return out


def draw_bounding_box(img, class_name, class_color, pt1, pt2, border_size=2):
    out = img
    color = class_color    
#    print(pt1, pt2, color)
    out = cv2.rectangle(out, (pt1[0], pt1[1]), (pt2[0], pt2[1]), color, border_size)
    if class_name is not None:
        label = str(class_name)
        out = cv2.putText(out, label, (pt1[0]-10,pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out
