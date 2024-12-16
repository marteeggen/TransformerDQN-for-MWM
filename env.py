import numpy as np
import gym
from gym import spaces
from skspatial.objects import Circle
from skspatial.objects import Line
import math

np.random.seed(1337)

class CircularEnvironment(gym.Env):
    def __init__(self, radius=10, platform_radius=0.75, num_sight_lines=12, field_of_view=1, rotation_angle=0.2, max_steps=500, num_landmarks=5, unique_colors=2):
        super(CircularEnvironment, self).__init__()
        self.radius = radius 
        self.platform_radius = platform_radius
        self.num_sight_lines = num_sight_lines
        self.field_of_view = field_of_view 
        self.rotation_angle = rotation_angle
        self.max_steps = max_steps
        self.num_landmarks = num_landmarks
        self.unique_colors = unique_colors
        self.steps_taken = 0

        self.agent_position = self.place_agent()
        self.agent_direction = -self.agent_position / np.linalg.norm(self.agent_position)

        self.invisible_platform = self.place_invisible_platform()
        self.landmark_positions = self.place_landmark_segments()
        self.landmark_colors = self.set_landmark_colors()

        # Observation space: 2n values (distance and color for each sight line)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2 * num_sight_lines,), dtype=np.float32)

        # Action space: 0 - no action, 1 - left turn, 2 - right turn, 3 - move forward
        self.action_space = spaces.Discrete(4)

    def place_invisible_platform(self):
        angle = np.random.uniform(0, 2*np.pi)
        distance = np.random.uniform(0.05*self.radius, self.radius - 0.5*self.radius)
        return np.array([distance * np.cos(angle), distance * np.sin(angle)])

    def place_agent(self):
        angle = np.random.uniform(0, 2*np.pi)
        distance = self.radius - 0.01*self.radius 
        return np.array([distance * np.cos(angle), distance * np.sin(angle)])

    def place_landmark_segments(self):
        angle_increment = 2 * math.pi / self.num_landmarks 

        segments = []
        for i in range(self.num_landmarks):
            theta_start = i * angle_increment
            theta_end = (i + 1) * angle_increment

            x1 = self.radius * math.cos(theta_start)
            y1 = self.radius * math.sin(theta_start)
            x2 = self.radius * math.cos(theta_end)
            y2 = self.radius * math.sin(theta_end)

            segments.append(((x1, y1), (x2, y2)))

        return segments

    def set_landmark_colors(self): 
        landmark_colors = {i: 0 for i in range(self.num_landmarks)}
        unique_colors = min(self.unique_colors, self.num_landmarks)
        step = self.num_landmarks // unique_colors
        for i in range(unique_colors):
            position = (i * step) % self.num_landmarks
            landmark_colors[position] = i + 1
        return landmark_colors

    def get_sight_lines(self):
        sight_lines = []
        intersection_points = []
        half_fov = self.field_of_view / 2

        for i in range(self.num_sight_lines):
            angle_offset = (i / (self.num_sight_lines - 1)) * self.field_of_view - half_fov
            distance, color, intersection_point = self.cast_ray(angle_offset)
            sight_lines.append((distance, color))
            intersection_points.append(np.array(intersection_point))
      
        return sight_lines, intersection_points

    def find_segment(self, point):
        x = point[0]
        y = point[1]

        angle = np.arctan2(y, x)

        if angle < 0:
            angle = angle + 2*np.pi

        angle_increment = 2 * math.pi / self.num_landmarks
        segment_index = np.floor(angle/angle_increment)
        return self.landmark_colors[segment_index]


    def cast_ray(self, angle_offset):
        direction_vector = self.rotate_vector(angle_offset) - self.agent_position

        line = Line(self.agent_position, direction_vector)
        circle = Circle([0, 0], self.radius) 

        all_intersection_points = circle.intersect_line(line)

        for point in all_intersection_points:
            if np.dot(direction_vector, np.array(point) - np.array(self.agent_position)) > 0:
                intersection_point = point

        distance = np.linalg.norm(np.array(self.agent_position) - np.array(intersection_point))
        color = self.find_segment(intersection_point)

        return distance, color, intersection_point


    def is_point_inside_circle(self, x1, y1, xc, yc, r):
        distance_squared = (x1 - xc) ** 2 + (y1 - yc) ** 2
        return distance_squared < r ** 2 
 

    def rotate_vector(self, angle): 
        v = self.agent_direction - self.agent_position

        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

        v = np.dot(rotation_matrix, v)
        v = v + self.agent_position
        
        return v 

    def step(self, action):

        sight_lines, _ = self.get_sight_lines()
        current_state_sight_lines = np.array(sight_lines).flatten()

        self.steps_taken += 1

        if self.steps_taken >= self.max_steps:
            return current_state_sight_lines, 0.0, True, {} 

        if action == 1: 
            self.agent_direction = self.rotate_vector(-self.rotation_angle)
          
        elif action == 2:  
            self.agent_direction = self.rotate_vector(self.rotation_angle)
            
        elif action == 3: 
            v = self.agent_direction - self.agent_position
            temp_position = self.agent_position + (v / np.linalg.norm(v))
            temp_direction = self.agent_direction + (v / np.linalg.norm(v)) 
    
            if not self.is_point_inside_circle(temp_position[0], temp_position[1], 0, 0, self.radius):
                return current_state_sight_lines, -0.3, False, {} 
            else:
                self.agent_position = temp_position
                self.agent_direction = temp_direction

        if self.is_point_inside_circle(self.agent_position[0], self.agent_position[1], self.invisible_platform[0], self.invisible_platform[1], self.platform_radius): 
            return current_state_sight_lines, 1.0, True, {} 

        sight_lines, _ = self.get_sight_lines()
        return np.array(sight_lines).flatten(), -0.0003, False, {} 
    
    def reset(self):
        self.agent_position = self.place_agent() 
        self.agent_direction = -self.agent_position / np.linalg.norm(self.agent_position)

        self.steps_taken = 0
        sight_lines, _ = self.get_sight_lines()
        return np.array(sight_lines).flatten()