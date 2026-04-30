from core import *
import math
import random

class EVChargingModel:
    def __init__(self, chargers = 4):
        self.chargers = chargers #Number of chargers
        self.busy = 0 #Number of chargers that is occupied
        self.queue = [] #EV's waiting for a charger
        self.in_service = [] #List of EV's currently charging
        self.arrivals = 0 #Total number EV's arrived
        self.reneged = 0 #number EV that are reneging
        self.completed = 0 #number EV's that have left the charging (departure or early departure)
        self.early_departure = 0 #number EV's leaving early
        self.total_waiting_time = 0 #sum of waiting times for EV's that start service
        self.area_queue = 0.0 #integral of queue length over time
        self.area_busy = 0.0 #integral of number of busy chargers over time

    def start_service(self, sim, ev):
        self.busy += 1 #set charger to busy
        self.in_service.append(ev) #put ev on currently being charged
        wait = sim.current_time - ev.arrival_time #calculate waiting time of ev
        self.total_waiting_time += wait #add waiting time to total waiting time

        if ev.reneging_event:
            sim.cancel(ev.reneging_event)
        end_time = sim.current_time + ev.service_time
        ev.service_event = sim.schedule(ServiceEndEvent(end_time, ev, self))

class EV:
    def __init__(self, n, arrival_time):
        self.n = n #index of arrival
        self.arrival_time = arrival_time #arrival time
        self.b = 0.5 * abs(math.sin(n * math.pi / 7) + 1) #battery amount
        self.service_time = 60 * (1 - self.b) #service time
        self.patience = 20 * (1 + abs(math.cos(n * math.e))) #patience of EV
        self.reneging_event = None #Did the EV reneging
        self.charging_event = None #Did the EV charge

def interarrival(n):
    return 15 * (1 + math.sin(n * math.pi / 12))**2 + 2 #calculate arrival time n+1 EV

class Arrival(Event):
    def __init__(self, time: float, n, model):
        super().__init__(time)
        self.n = n
        self.model = model

    def execute(self, sim):
        m = self.model
        m.arrivals +=1 #set counter plus 1
        ev = EV(self.n, sim.current_time) #create ev
        next_time = sim.current_time+interarrival(self.n) #schedule next arrival
        sim.schedule(Arrival(next_time, self.n + 1, m))

        if m.busy < m.chargers: #see if charger is available
            m.start_service(sim, ev) #start charging
        else:
            m.queue.append(ev) #add ev to queue
            reneging_time = sim.current_time + ev.patience #determine reneging moment
            ev.reneging_event = sim.schedule(Reneging(reneging_time, ev, m)) #schedule reneging

            if len(m.queue) % 5 == 0 and len(m.queue) != 0: #check if queue is multiple of 5
               trigger_early_departure(sim, m) #check the early departure

class ServiceEndEvent(Event):
    def __init__(self, time, ev, model):
        super().__init__(time)
        self.ev = ev
        self.model = model

    def execute(self, sim):
        m = self.model
        m.busy -= 1 #set one busy charger to free
        m.in_service.remove(self.ev)  #remove ev from curently being charged
        m.completed += 1 #increase the conpleted counter

        if m.completed >= 800: #stop condition
            sim.stop()
            return

        if m.queue: #check if queue is not empty
            next_ev = m.queue.pop(0) #remove next ev from queue
            m.start_service(sim, next_ev) #start service next ev

class Reneging(Event):
    def __init__(self, time, ev, model):
        super().__init__(time)
        self.ev = ev
        self.model = model

    def execute(self, sim):
        m = self.model

        if self.ev in m.queue: #check if ev is in queue
            m.queue.remove(self.ev) #remove ev from queue
            m.reneged +=1 #increase counter by one

def trigger_early_departure(sim, m):
    for ev in list(m.in_service): #check all ev that are currently being charged
        remaining = ev.service_event.time - sim.current_time #caculate the time remaining

        if remaining > 15: #check if the remaining time is bigger then 15
            if random.random() < 0.2: #with 20 precent check to interrupt
                sim.cancel(ev.service_event) #cancel the charging
                sim.schedule(EarlyDeparture(sim.current_time + 2, ev, m)) #schedule early departure in 2 minutes

class EarlyDeparture(Event):
    def __init__(self, time, ev, model):
        super().__init__(time)
        self.ev = ev
        self.model = model

    def execute(self, sim):
        m = self.model

        if self.ev in m.in_service:
            m.in_service.remove(self.ev) #remove ev from system
            m.busy -= 1 #decrease busy chargers with one
            m.completed += 1 #increase the number of cars the have been charged
            m.early_departure += 1 #increase the counter for early departures

            if m.completed >= 800: #stop condition
                sim.stop()
                return

            if m.queue: #check if queue is not empty
                next_ev = m.queue.pop(0) #remove next ev from queue
                m.start_service(sim, next_ev) #start service for that next ev

def time_update(sim, event):
    m = sim.model
    dt = sim.current_time - sim.previous_time #determine time between two decision moments

    m.area_queue += len(m.queue) * dt #length of queue in between each two decision moments times the time of this period
    m.area_busy += m.busy * dt #number of busy chargeer in between two decision moements times this time

#running simulation
if __name__ == '__main__':
    sim = Simulation()
    model = EVChargingModel()

    sim.model = model
    sim.on_before_event(time_update)

    sim.schedule(Arrival(0.0, 1, model))
    sim.run()
    T = sim.current_time

    average_queue = model.area_queue / T
    average_wait = model.total_waiting_time / (model.completed)
    reneging_fraction = model.reneged / model.arrivals
    utilisation = model.area_busy / (model.chargers * T)
    early_departure_fraction = model.early_departure / model.completed

    print("Average queue length:", average_queue)
    print("Average waiting time:", average_wait)
    print("Reneging fraction:", reneging_fraction)
    print("Utilisation:", utilisation)
    print("Early departure fraction:", early_departure_fraction)
