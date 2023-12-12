import ast
import copy
from multiprocessing import Pool, cpu_count

import numpy as np
from math import dist
import pandas as pd


class VRP:
    """VRP (Vehicle Routing Problem) is a class to minimize cost in routing trucks to locations"""

    def __init__(self, max_trips: int, random_walks: int):
        """
        :param max_trips: max_trips is the maximum amount of trips in a single driver can take
        :param random_walks: number of random walks to take in the cartesian space
        """
        self.max_trips = max_trips
        self.random_walks = random_walks

    def run(self, orders: pd.DataFrame) -> list:
        """run will route trucks and minimize cost
        :param orders: pd.DataFrame with 3 columns - loadNumber, pickup, dropoff
        :return: list of orders for a single truck driver
        """
        self.orders = self.str_to_val(orders)
        self.dist_mat = self.distance_matrix(orders)

        return self.minimize_cost()

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

    def minimize_cost(self) -> list:
        """minimize_cost determines best orders to couple with the same driver
        :return: list of best pairs
        """
        scores, tracks = [], []
        p = Pool(8)
        for idx in range(self.dist_mat[0].shape[0]):
            scores = p.map(self.random_search, [idx for _ in range(self.random_walks)])
        
        track_scores, tracks = [s for s, _  in scores], [t for _, t in scores]
        p.close(), p.join()
        return self.find_orders(track_scores, tracks)
    

    def random_search(self, start_idx: int) -> tuple:
        """ random_search takes random walks in the cartesian space
        :param start_idx: idx corresponding to location
        :return (float score, indices of track)
        """
        rand_track = [start_idx]

        size = np.random.randint(2, self.max_trips)

        while start_idx in rand_track:
            rand_track = np.arange(0, self.dist_mat[0].shape[0])
            np.random.shuffle(rand_track)
            rand_track = rand_track[:size]

        rand_track = np.append([start_idx], rand_track)

        score = self.total_cost(rand_track)
        return (score, rand_track)

    def total_cost(self, order_nums: list) -> float:
        """total_cost uses a paths cost per each order completed and calcuates total cost
        :param order_nums: list of loadNumbers
        :return: calculates global cost for the order
        """
        first_loc, *next_locations = order_nums
        cost = self.dist_mat[first_loc, first_loc]

        current_loc = copy.copy(first_loc)
        for loc in next_locations:
            cost += self.dist_mat[current_loc, loc] + self.dist_mat[loc, loc]
            current_loc = loc

        score = (
            dist((0, 0), self.orders.iloc[first_loc, 1])
            + cost
            + dist(self.orders.iloc[current_loc, 2], (0, 0))
        )

        return score if score < 60*12 else np.inf

    def find_orders(self, scores: list, tracks: list) -> list:
        """find_orders finds the best orders with minimal scores
        :param scores: list of scores
        :param tracks: list of tracks
        :return: list of tracks
        """

        seen_orders, final_tracks = np.array([]), []
        normalized_scores = [self.normalized_scores(s, t) for s, t in zip(scores, tracks)]
        best_scores = np.array(normalized_scores).argsort()

        for idx in best_scores:
            if scores[idx] > 12*60:
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
    
    def normalized_scores(self, score, track):
        """normalized_scores reduces scores that have multiple points with large distances from the origin
        :param scores: float score
        :param track: list of places visited
        :return: new float score
        """
        if score == np.inf: return score
        
        distance = 0
        for place in track:
            distance += max(
                dist(self.orders['pickup'][place], (0,0)), 
                dist(self.orders['dropoff'][place], (0,0))
                )
        
        return score / distance
