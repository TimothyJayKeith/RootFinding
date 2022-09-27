from array import array
import numpy as np

def NewUnitArea(bounding_box, old_area=0):
    """
    A function that finds the area of a new bounding box in the [-1, 1]^n rectangle
    in relation to the original.
    
    Parameters
    ----------
    bounding_box: A 2xn dimensional numpy array
        The first row of the array is the lower bounds of each of the dimensions of the box
        while the second row is the upper bounds of each of these dimensions
    old_area: a float between 0 and 1
        This is the area of the previous box in the interval shrink, used to find the area of
        the current box with respect to the original box

    Returns
    -------
    The area of the current box with respect to the original
    """
    return (np.prod(bounding_box[1] - bounding_box[0])/2**(len(bounding_box[0])))*old_area

def NewLogArea(bounding_box, old_area=0.0):
    """
    A function that finds the log of the area of a new bounding box in the [-1, 1]^n rectangle
    in relation to the original.
    
    Parameters
    ----------
    bounding_box: A 2xn dimensional numpy array
        The first row of the array is the lower bounds of each of the dimensions of the box
        while the second row is the upper bounds of each of these dimensions
    old_area: a float between 0 and 1
        This is the area of the previous box in the interval shrink, used to find the area of
        the current box with respect to the original box

    Returns
    -------
    The log of the area of the current box with respect to the original
    """
    return np.sum(np.log(bounding_box[1] - bounding_box[0])) - len(bounding_box[0])*np.log(2) + old_area

def SumLogAreas(bounding_boxes, old_log_areas=[]):
    """
    Finds the log of a the sum of the areas of different boxes.

    Parameters
    ----------
    bounding_boxes: a list (or other iterable)
        A list of different bounding boxes, entered as an array of the box's extreme values
    old_area_sum: float
        The previous sum of areas in an iteration

    Returns
    -------
    The log of the sum of the areas of the current boxes
    """
    if len(old_log_areas) == 0:
        old_log_areas = [0.0] * len(bounding_boxes)
    new_log_areas = [NewLogArea(bounding_boxes[i], old_log_areas[i]) for i in range(len(bounding_boxes))]
    log_areas = new_log_areas.copy()
    max_log_area = log_areas.pop(log_areas.index(max(log_areas)))
    array_to_sum = np.array([np.exp(log_area - max_log_area) for log_area in log_areas])
    return max_log_area + np.log(1 + np.sum(array_to_sum)), new_log_areas

def TrackProgress(area, dim, precision=1e-15):
    """
    
    """
    return area/(dim*(np.log(precision) - np.log(2)))