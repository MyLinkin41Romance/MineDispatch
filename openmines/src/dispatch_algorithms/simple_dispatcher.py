from __future__ import annotations
from openmines.src.dispatcher import BaseDispatcher
from openmines.src.load_site import LoadSite
import numpy as np

class SimpleDispatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "SimpleDispatcher"

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        return 0

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        current_location = truck.current_location
        assert isinstance(current_location, LoadSite), "current_location is not a LoadSite"
        cur_index = mine.load_sites.index(current_location)

        distances = [np.linalg.norm(np.array(current_location.position) - np.array(dump_site.position)) 
                     for dump_site in mine.dump_sites]
        nearest_dump_index = distances.index(min(distances))
        return nearest_dump_index

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        return 0