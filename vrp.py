import ast
import copy

import numpy as np
from math import dist
import pandas as pd


class VRP:
    """VRP (Vehicle Routing Problem) is a class to minimize cost in routing trucks to locations"""

    def __init__(self, max_trips: int, cost_per_driver: int):
        """
        :param max_trips: max_trips is the maximum amount of trips in a single driver can take
        """
        self.max_trips = max_trips
        self.cost = 0
        self.cost_per_driver = cost_per_driver

    def run(self, orders: pd.DataFrame) -> list:
        """run will route trucks and minimize cost
        :param orders: pd.DataFrame with 3 columns - loadNumber, pickup, dropoff
        :return: list of orders for a single truck driver
        """
        self.orders = self.str_to_val(orders)
        dist_mat = self.distance_matrix(orders)

        return self.minimize_cost(dist_mat=dist_mat)

    def distance_matrix(self, orders: list) -> np.ndarray:
        """distance_matrix finds distances between all orders (dropoff -> pickup)
        :param orders: pd.DataFrame with 3 columns - loadNumber, pickup, dropoff
        :return: np
        """
        num_orders = len(orders)
        dist_mat = np.zeros(shape=(num_orders, num_orders))
        points = np.array([orders["pickup"], orders["dropoff"]]).T

        for x_idx, (x_pick, x_drop) in enumerate(points):
            for y_idx, (y_pick, y_drop) in enumerate(points):
                if x_idx == y_idx:
                    dist_mat[x_idx, y_idx] = self.cost_fn(x_pick, x_drop)
                dist_mat[x_idx, y_idx] = self.cost_fn(x_drop, y_pick)

        return dist_mat

    def cost_fn(self, x: tuple, y: tuple) -> float:
        """cost_fn is euclidean distance of (tuple, tuple)
        :param x: tuple with (pick-up, drop-off)
        :param y: tuple with (pick-up, drop-off)
        :return: total distance between x, y
        """
        return dist(x, y)

    def str_to_val(self, orders: pd.DataFrame) -> pd.DataFrame:
        """str_to_val processes a string to a value
        :param orders: DataFrame with str values
        :return: DataFrame with tuples
        """
        orders["pickup"] = orders["pickup"].map(lambda x: ast.literal_eval(x))
        orders["dropoff"] = orders["dropoff"].map(lambda x: ast.literal_eval(x))

        return orders

    def minimize_cost(self, dist_mat: np.ndarray) -> list:
        """minimize_cost determines best orders to couple with the same driver
        :param dist_mat: distance matrix for each order coupled together
        :return: list of best pairs
        """
        scores, tracks = [], []
        for idx in range(dist_mat[0].shape[0]):
            rand_track = [idx]
            for _ in range(1000):
                size = np.random.randint(1, self.max_trips)

                while idx in rand_track:
                    rand_track = np.arange(0, dist_mat[0].shape[0])
                    np.random.shuffle(rand_track)
                    rand_track = rand_track[:size]

                rand_track = np.append([idx], rand_track)

                score = self.total_cost(rand_track, dist_mat)
                scores.append(score), tracks.append(rand_track)

        return self.find_orders(scores, tracks)

    def total_cost(self, order_nums: list, dist_mat: np.ndarray) -> float:
        """total_cost uses a paths cost per each order completed and calcuates total cost
        :param order_nums: list of loadNumbers
        :param dist_mat: distance matrix for each order coupled together
        :return: calculates global cost for the order
        """
        first_loc, *next_locations = order_nums
        cost = dist_mat[first_loc, first_loc]

        current_loc = copy.copy(first_loc)
        for loc in next_locations:
            cost += dist_mat[current_loc, loc] + dist_mat[loc, loc]
            current_loc = loc

        return (
            dist((0, 0), self.orders.iloc[first_loc, 1])
            + cost
            + dist(self.orders.iloc[current_loc, 2], (0, 0))
        )

    def find_orders(self, scores: list, tracks: list) -> list:
        """find_orders finds the best orders with minimal scores
        :param scores: list of scores
        :param tracks: list of tracks
        :return: list of tracks
        """

        seen_orders, final_tracks = np.array([]), []
        best_scores = np.array(scores).argsort()

        for idx in best_scores:
            if scores[idx] > 720:
                break

            track = (
                np.array(tracks[idx]) + 1
            )  # these are indices and need mapped to loadNumbers
            if any(x in seen_orders for x in track):
                continue

            seen_orders = np.append(seen_orders, track)
            final_tracks.append(track.tolist())

        for x in range(1, len(self.orders) + 1):
            if x in seen_orders:
                continue
            final_tracks.append([x])

        return final_tracks
