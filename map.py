# Self Driving Car Game

import numpy
from random import random, randint
import matplotlib.pyplot as plt
import time

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

from ai import DeepQNetwork

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Records last block of sand when drawing lines of sand
last_x = 0
last_y = 0
n_points = 0
length = 0

# Initializes the DeepQNetwork(input_size, number_of_actions, gamma)
brain = DeepQNetwork(5, 3, 0.9)

# Angle at which the different actions will take
rotations_of_actions = [0, 20, -20]

# Records the reward the car got at its last state
last_reward = 0

# Stores the rewards
scores = []

first_update = True

def init():
    # Goal destination coordinates
    global goal_x
    global goal_y
    goal_x = 20
    goal_y = width - 20

    global first_update
    first_update = False

    # Array where 1 represents a sand block and 0 represents a non-sand block (all 0 by default)
    global sand
    sand = numpy.zeros((length, width))


# Current distance between the car and the goal destination (0 by default)
last_distance = 0


# Creating the car class
class Car(Widget):

    # Angle between x-axis and the direction of the car
    angle = NumericProperty(0)

    # Angle of the last rotation taken by the car (one of the angles in rotations_of_actions)
    rotation = NumericProperty(0)

    # Velocity coordinates and vector
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    # Detecting obstacles in front of the car
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)

    # Detecting obstacles on the left of the car
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)

    # Detecting obstacles on the right of the car
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)

    # Signals are the reward of the AI
    # A signal of 0 is good and a signal of 1 is bad
    # The value is calculated using the density of sand (side of the road) around each sensor
    # The greater the density of sand, the closer the signal is to 1 (bad)
    # If the car is close to the edge of the map, it is automatically set to 1
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        # Updates the current position using the velocity vector
        self.pos = Vector(*self.velocity) + self.pos

        # Rotates the current angle of the car using the rotation of the action
        self.rotation = rotation
        self.angle = self.angle + self.rotation

        # Updates the position and angle of the sensors (30 is the distance between the car and the sensors)
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos

        # Caculates the new sand density around each sensor
        self.signal1 = int(numpy.sum(sand[int(self.sensor1_x) - 10:int(self.sensor1_x) + 10, int(self.sensor1_y) - 10:int(self.sensor1_y) + 10])) / 400.
        self.signal2 = int(numpy.sum(sand[int(self.sensor2_x) - 10:int(self.sensor2_x) + 10, int(self.sensor2_y) - 10:int(self.sensor2_y) + 10])) / 400.
        self.signal3 = int(numpy.sum(sand[int(self.sensor3_x) - 10:int(self.sensor3_x) + 10, int(self.sensor3_y) - 10:int(self.sensor3_y) + 10])) / 400.

        # Checks if the car is close to the edge of the map and sets the sensor to 1 if it is
        if self.sensor1_x > length - 10 or self.sensor1_x < 10 or self.sensor1_y > width - 10 or self.sensor1_y < 10:
            self.signal1 = 1.
        if self.sensor2_x > length - 10 or self.sensor2_x < 10 or self.sensor2_y > width - 10 or self.sensor2_y < 10:
            self.signal2 = 1.
        if self.sensor3_x > length - 10 or self.sensor3_x < 10 or self.sensor3_y > width - 10 or self.sensor3_y < 10:
            self.signal3 = 1.


# Graphical balls that represent the sensors on the map
class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass


# Creates the game class
class Game(Widget):

    car = ObjectProperty(None)

    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global length
        global width

        length = self.width
        width = self.height

        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y

        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.

        # These are the inputs in the neural network
        # -orientation seems redundant, but it will allow more exploration by our AI
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]

        action = brain.update(last_reward, last_signal)

        # Adds the last score to the list of rewards
        scores.append(brain.score())

        # Updates the last rotation taken by the car with the new rotation (decided by the neural network)
        rotation = rotations_of_actions[action]

        # Moves the car with the new rotation
        self.car.move(rotation)

        # Updates the distance from the car to the goal destination
        distance = numpy.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)

        # Updates the locations of the sensors on the map
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        # Checks if car hit sand (side of the road)
        # If it did, it slows the car to speed 1 and gives a negative reward of -1
        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1
        else:
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)

            # Gives a temporary negative reward that will be changed to a positive one if the car is closer to the goal destination
            last_reward = -0.2

            # Gives a 0.1 reward if the car is closer to the goal destination than in the last state
            if distance < last_distance:
                last_reward = 0.1

        # Gives a negative reward of -1 and moves the car if it hit any edge of the map
        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        # Sets the new goal to be the initial starting point if the car has reached the goal destination
        if distance < 100:
            goal_x = self.width - goal_x
            goal_y = self.height - goal_y

        # Updates the last distance between the car and the goal destination
        last_distance = distance


# Adds the painting tools for Kivy
class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8, 0.7, 0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x), int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += numpy.sqrt(max((x - last_x) ** 2 + (y - last_y) ** 2, 2))
            n_points += 1.
            density = n_points / (length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y


# Adds the GUI Buttons (clear, save and load)
class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = numpy.zeros((length,width))

    # Saves the AI
    def save(self, obj):
        print("Saving the AI...")
        brain.save()
        plt.plot(scores)
        plt.show()

    # Loads the AI
    def load(self, obj):
        print("Loading the last saved AI...")
        brain.load()

# Runs the program
if __name__ == '__main__':
    CarApp().run()
