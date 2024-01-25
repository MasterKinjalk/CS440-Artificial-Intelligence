# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy
import math


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    alien_width = alien.get_width()
    is_circle = alien.is_circle()

    for wall in walls:
        wall_segment = ((wall[0], wall[1]), (wall[2], wall[3]))
        dist = point_segment_distance(alien.get_centroid(), wall_segment)

        if is_circle and dist <= alien_width:
            return True

        if not is_circle:
            head,tail = alien.get_head_and_tail()
            if do_segments_intersect((head,tail), wall_segment) or segment_distance((head,tail), wall_segment) -alien_width <= 0:
                return True

    return False

def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    width, height = window
    alien_width = alien.get_width()
    is_circle = alien.is_circle()

    if is_circle:
        x, y = alien.get_centroid()
        return alien_width <= x <= width - alien_width and alien_width <= y <= height - alien_width
    else:
        (x1, y1), (x2, y2) = alien.get_head_and_tail()
        if alien.get_shape() == "Horizontal":
            return alien_width <= min(x1, x2) and max(x1, x2) + alien_width <= width and alien_width <= y1 <= height - alien_width
        else:  # "Vertical"
            return alien_width <= min(y1, y2) and max(y1, y2) + alien_width <= height and alien_width <= x1 <= width - alien_width

def is_point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    on_boundary = False

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        if y > min(y1, y2) and y <= max(y1, y2) and x <= max(x1, x2) and y1 != y2:
            x_intersect = (y - y1) * (x2 - x1) / (y2 - y1) + x1

            if x1 == x2 or x <= x_intersect:
                on_boundary = not on_boundary
            if x == x_intersect:
                return True  # Point lies on the edge

    return on_boundary


def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    alien_width = alien.get_width()
    alien_length = alien.get_length()
    alien_shape = alien.get_shape()
    head, tail = alien.get_head_and_tail()

    if(alien.is_circle()):
        centroid = alien.get_centroid()
        pathSegment = (centroid, waypoint)
        for wall in walls:
            wallPoints = ((wall[0], wall[1]), (wall[2], wall[3]))
            if segment_distance(pathSegment, wallPoints) <= alien_width:
                return True
    else:
        if(alien_shape == 'Horizontal'):
            waypoint_head = (waypoint[0]+alien_length/2, waypoint[1])
            waypoint_tail = (waypoint[0]-alien_length/2, waypoint[1])
            head,tail = tail,head
        else:  # Vertical
            waypoint_head = (waypoint[0], waypoint[1]+alien_length/2)
            waypoint_tail = (waypoint[0], waypoint[1]-alien_length/2)
            head,tail = head,tail
        for wall in walls:
            if(is_point_in_polygon((wall[0],wall[1]),(head,tail,waypoint_tail,waypoint_head)) or is_point_in_polygon((wall[2],wall[3]),(head,tail,waypoint_tail,waypoint_head))):
                return True
            
            wallPoints = ((wall[0],wall[1]),(wall[2],wall[3]))
            if any([segment_distance(wallPoints, (head,tail)) <= alien_width , segment_distance(wallPoints, (waypoint_head,waypoint_tail)) <= alien_width , segment_distance(wallPoints, (tail,waypoint_head)) <= alien_width , segment_distance(wallPoints, (head,waypoint_tail)) <= alien_width ]):
                return True

    return False


import numpy as np

def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    p = np.array(p)
    s = np.array(s)
    
    v = s[1] - s[0]
    w = p - s[0]

    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(p - s[0])

    c2 = np.dot(v, v)
    if c2 <= c1:
        return np.linalg.norm(p - s[1])

    b = c1 / c2
    pb = s[0] + b * v
    return np.linalg.norm(p - pb)

#This logic has been inferred from a GFG post at: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
def do_segments_intersect(s1, s2):
    X1, Y1 = s1[0]
    X2, Y2 = s1[1]
    #print(s2[0])
    X3, Y3 = s2[0]
    X4, Y4 = s2[1]
    
    def space_orientation(X1, Y1, X2, Y2, X3, Y3):
        val = (Y2 - Y1) * (X3 - X2) - (X2 - X1) * (Y3 - Y2)
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    o1 = space_orientation(X1, Y1, X2, Y2, X3, Y3)
    o2 = space_orientation(X1, Y1, X2, Y2, X4, Y4)
    o3 = space_orientation(X3, Y3, X4, Y4, X1, Y1)
    o4 = space_orientation(X3, Y3, X4, Y4, X2, Y2)

    if (o1 != o2 and o3 != o4) or \
       (o1 == 0 and on_line_segment(X1, Y1, X3, Y3, X2, Y2)) or \
       (o2 == 0 and on_line_segment(X1, Y1, X4, Y4, X2, Y2)) or \
       (o3 == 0 and on_line_segment(X3, Y3, X1, Y1, X4, Y4)) or \
       (o4 == 0 and on_line_segment(X3, Y3, X2, Y2,X4,Y4)):
        return True

    return False


def on_line_segment(X1,Y1,X2,Y2,X3,Y3):
    is_it_on_line = (Y2 <= max(Y1,Y3) and 
                     Y2 >= min(Y1,Y3) and 
                     X2 <= max(X1,X3) and 
                     X2 >= min(X1,X3))

    return is_it_on_line




def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    # If the segments intersect, the distance is zero
    if do_segments_intersect(s1, s2):
        return 0

    # Calculate the minimum distance between each endpoint of one segment and the other segment
    distance = min([point_segment_distance(s1[0], s2), point_segment_distance(s1[1], s2),
                 point_segment_distance(s2[0], s1), point_segment_distance(s2[1], s1)])

    # Return the minimum distance
    return distance



if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
