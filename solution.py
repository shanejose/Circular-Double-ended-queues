"""
CSE331 Project 6 SS'23
Circular Double-Ended Queue
solution.py
"""

from typing import TypeVar, List
from random import randint, shuffle
from timeit import default_timer
# from matplotlib import pyplot as plt  # COMMENT OUT THIS LINE (and `plot_speed`) if you dont want matplotlib
import gc

T = TypeVar('T')


class CircularDeque:
    """
    Representation of a Circular Deque using an underlying python list
    """

    __slots__ = ['capacity', 'size', 'queue', 'front', 'back']

    def __init__(self, data: List[T] = None, front: int = 0, capacity: int = 4):
        """
        Initializes an instance of a CircularDeque
        :param data: starting data to add to the deque, for testing purposes
        :param front: where to begin the insertions, for testing purposes
        :param capacity: number of slots in the Deque
        """
        if data is None and front != 0:
            data = ['Start']  # front will get set to 0 by a front enqueue if the initial data is empty
        elif data is None:
            data = []

        self.capacity: int = capacity
        self.size: int = len(data)
        self.queue: List[T] = [None] * capacity
        self.back: int = (self.size + front - 1) % self.capacity if data else None
        self.front: int = front if data else None

        for index, value in enumerate(data):
            self.queue[(index + front) % capacity] = value

    def __str__(self) -> str:
        """
        Provides a string representation of a CircularDeque
        'F' indicates front value
        'B' indicates back value
        :return: the instance as a string
        """
        if self.size == 0:
            return "CircularDeque <empty>"

        str_list = ["CircularDeque <"]
        for i in range(self.capacity):
            str_list.append(f"{self.queue[i]}")
            if i == self.front:
                str_list.append('(F)')
            elif i == self.back:
                str_list.append('(B)')
            if i < self.capacity - 1:
                str_list.append(',')

        str_list.append(">")
        return "".join(str_list)

    __repr__ = __str__

    #
    # Your code goes here!
    #
    def __len__(self) -> int:
        """
        Return the length/size of the circular deque

        @return: int, the length of the circular dequeue
        """

        return self.size

    def is_empty(self) -> bool:
        """
        Return a boolean indicating if the circular deque is empty

        @return: bool, True if empty, False otherwise
        """

        return self.size == 0

    def front_element(self) -> T:
        """
        Return the front element in the circular deque

        @return: T, the first element if it exists, otherwise None
        """
        if self.size == 0:
            return None

        return self.queue[self.front]

    def back_element(self) -> T:
        """
        Return the last element in the circular deque

        @return: T, the last element if it exists, otherwise None
        """
        if self.size == 0:
            return None

        return self.queue[self.back]

    def enqueue(self, value: T, front: bool = True) -> None:
        """
        Add a value to the circular deque based off the parameter front
        If front is True, add the value to front of the circular deque
        If front is False, add the value to back of the circular deque
        Increase the size of the list if the list has reached capacity

        @param value: T, value to add into the circular deque
        @param front: bool, where to add value T
        @return None
        """

        if self.is_empty():
            self.front = 0
            self.back = 0
        elif front:
            self.front = (self.front - 1) % self.capacity

        else:
            self.back = (self.back + 1) % self.capacity

        if front:
            self.queue[self.front] = value
        else:
            self.queue[self.back] = value
        self.size += 1

        if self.size == self.capacity:
            self.grow()

    def dequeue(self, front: bool = True) -> T:
        """
        Remove an item from the queue
        Remove from the front by default, remove the back item if False is passed in for front

        @param front: bool, Whether to remove the front or back item from the dequeue
        @return T, removed item, None if empty
        """
        if self.is_empty():
            return None

        if front:
            removed_item = self.queue[self.front]
            self.front = (self.front + 1) % self.capacity
        else:
            removed_item = self.queue[self.back]
            self.back = (self.back - 1) % self.capacity

        self.size -= 1

        if self.capacity > 4 and self.size <= self.capacity // 4:
            self.shrink()

        return removed_item

    def grow(self) -> None:
        """
        Double the capacity of the circular deque
        Create a new list with double the capacity of the old one
        Copy the values over from the current list to list with new capacity

        @return None
        """

        new_capacity = self.capacity * 2
        new_queue = [None] * new_capacity

        for i in range(self.size):
            new_queue[i] = self.queue[(self.front + i) % self.capacity]

        self.queue = new_queue
        self.capacity = new_capacity
        self.front = 0
        self.back = self.size - 1

    def shrink(self) -> None:
        """
        Cuts the capacity of the queue in half using the same idea as grow
        If size <= 1/4 current capacity, and 1/2 current capacity >= 4, halves the capacity

        @return None
        """
        if self.capacity <= 4:
            return

        new_capacity = max(self.capacity // 2, 4)
        new_queue = [None] * new_capacity

        # Copy over contents to new queue
        for i in range(self.size):
            new_queue[i] = self.queue[(self.front + i) % self.capacity]

        # Update the queue and capacity
        self.queue = new_queue
        self.capacity = new_capacity
        self.front = 0
        self.back = self.size - 1


def get_winning_numbers(numbers: List[int], size: int) -> List[int]:
    """
    Returns a list containing the maximum value of the sliding window at each iteration step

    @param numbers: list, a list of numbers that the sliding window will move through
    @param size: int, the size of the sliding window
    @return max_value: list, list containing the max sliding window at each iteration step
    """
    if not numbers or size == 0:
        return []

    if size == 1:
        return numbers

    result = []
    circular_deque = CircularDeque()

    left_pointer = 0
    right_pointer = 0

    while right_pointer < len(numbers):
        # Dequeues elements from the back of the dequeue as long as it's smaller than the current element
        while not circular_deque.is_empty() and numbers[circular_deque.back_element()] < numbers[right_pointer]:
            circular_deque.dequeue(front=False)
        circular_deque.enqueue(right_pointer, front=False)

        # Check if we are within the current sliding window
        if left_pointer > circular_deque.front_element():
            circular_deque.dequeue(front=True)

        # Check if window has reached the required size
        if right_pointer + 1 >= size:
            # append the front item to result because it is the maximum element in the window
            result.append(numbers[circular_deque.front_element()])
            left_pointer += 1
        right_pointer += 1

    return result


def get_winning_probability(winning_numbers: List[int]) -> int:
    """
    Takes in a list of winning numbers and returns the probability of the numbers winning
    By finding the largest sum of non-adjacent numbers

    @param winning_numbers: list, list of winning numbers
    @return: max_val: int, integer representing the probability of the numbers winning
    """

    if not winning_numbers:
        return 0

    max_val = 0
    cur_item = 0

    # Calculates maximum possible prize money that can be won if that element is selected
    for n in winning_numbers:
        # sum of current_element and second highest possible prize money that can be won
        prob = max(n + cur_item, max_val)
        cur_item = max_val
        max_val = prob

    return max_val



