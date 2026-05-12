from core import *
import random
import numpy as np
from scipy.stats import t

class CityWasteModel:
    def __init__(self, N = 3, M = 3, K = 5, p = 0.5):
        self.N = N  #numbers of districts
        self.M = M  #number of trucks
        self.K = K  #Rerouting value
        self.p = p  #probability for bernoulli process

        #parameters
        self.lambdas = [0.4, 0.4, 0.4]  #poisson arrival rates
        self.q1 = 1/3   #waste probabilities
        self.q2 = 1/3
        self.q3 = 1 - self.q1 - self.q2
        self.k1 = 2 #parameters for service districution
        self.k2 = 3
        self.mu1 = 1.0
        self.mu2 = 1.5
        self.mu3 = 1.0

        self.queues = []    #make a queue for all districts
        for district in range(N):
            self.queues.append([])

        self.trucks = []    #initialize a list of trucks with properties of the truck
        for j in range(M):
            truck = {
                "id": j,
                "home" : j, #home district
                "location": j,  #current location
                "status": "idle",   #whether truck is idle or busy
                "current_request": None,    #current request that the truck is working on
                "completion_time": None} #time when completion is scheduled
            self.trucks.append(truck)

        #Statistical counters
        self.last_event_time = 0.0  #time of last event

        self.area_queue = [0.0]*N   #queue length over time per district
        self.area_busy = [0.0]*M    #busy time of truck per district
        self.total_waiting_time = [0.0]*N   #waiting time per district
        self.arrivals = [0]*N   #number of arrivals
        self.completed = 0  #number of completed request
        self.reroutings = 0 #number of reroutings

        self.cycle_start_time = 0.0 #start time of a cycle
        self.cycle_waiting_sum = [0.0]*N    #waiting times per disctrict in a cycle
        self.cycle_queue_area = [0.0]*N #queue length over time of a cycle per district
        self.cycle_busy_area = [0.0]*M  #busy time in a cycle per district
        self.cycle_completed = [0.0]*N  #number of completed requests in a cycle
        self.cycle_arrivals = [0]*N #number of arrivals in a cycle

        self.cycle_time = []    #how long a cycle takes
        self.cycle_waiting_hist = []    #stored waiting times of all cycles
        self.cycle_queue_hist = []  #stored queuelengths of all cycles
        self.cycle_busy_hist = []   #stored utilization of all cycles
        self.cycle_completed_hist = []  #stored completed request per cycle
        self.cycle_arrivals_hist = []   #stored number of arrivals in a cycle

    def update_time_stats(self, sim):   #called at every event to update the times
        dt = sim.current_time - self.last_event_time    #time interval between two events
        self.last_event_time = sim.current_time #update last event time

        for district in range(self.N):  #compute queue lengths
            q = len(self.queues[district])  #all request waiting
            for truck in self.trucks:   #all request beining served
                if truck["status"] == "busy" and truck["location"] == district:
                    q += 1
            self.area_queue[district] += q * dt #updating queue length over time
            self.cycle_queue_area[district] += q * dt   #updating queue length over time for certain cycle

        for j, truck in enumerate(self.trucks): #compute utility of truck
            if truck["status"] == "busy":   #for all trucks that are busy update
                self.area_busy[j] += dt #update utilization (time truck is busy)
                self.cycle_busy_area[j] += dt   #update utilization per cycle

    def sample_service_time(self, waste_type): #determine servicetime (wastetype)
        if waste_type == 1: #if wastetype is organic
            service = np.random.gamma(self.k1, 1/self.mu1)
        elif waste_type == 2:   #if wastetype is recyclable
            service = np.random.gamma(self.k2, 1/self.mu2)    #if wastetype is general
        else:
            service = np.random.exponential(1/self.mu3)
        return service

    def try_dispatch(self, sim):    #idle truck start truck decision process
        for j, truck in enumerate(self.trucks):
            if truck["status"] == "idle":
                self.truck_decision(j, sim)

    def truck_decision(self, j, sim):   #truck decides which request to serve
        truck = self.trucks[j]  #get truck j from list of trucks

        #first we check if there are request at home
        home = truck["home"]    #check home of the truck
        if self.queues[home]:
            self.start_service(j, home, sim)    #start service at home
            return

        #check all other districts
        for k in range(1, self.N): #loop through all other districts
            district = (home + k) % self.N  #((i + 1) mod N ) + 1 dictrict
            if self.queues[district]:   #check if request in queue
                if random.random() < self.p: #bernoulli trail
                    self.start_service(j,district, sim) #if succes start trail
                    return  #failure try next

    def start_service(self, truck_id, district, sim):
        truck = self.trucks[truck_id]   #get truck id
        request = self.queues[district].pop(0)  #remove the request from queue

        #updates truck statuses
        truck["status"] = "busy"    #set truck to busy
        truck["location"] = district    #link district to truck
        truck["current_request"] = request  #link request to truck

        service_time = self.sample_service_time(request["type"]) #get service time
        completion_time = sim.current_time + service_time   #determine completin time
        event = ServiceCompletion(completion_time, truck_id, district, model)   #create completion event
        truck["completion_event"] = event   #add completion event to truck

        sim.schedule(event) #schedule completion

    def is_regeneration_state(self):    #check if system is empty (hence new cycle)
        for queue in self.queues:   #checks for empty queues
            if len(queue) > 0:
                return False
        for truck in self.trucks:   #checks for idle trucks
            if truck["status"] == "busy":
                return False
        return True

    def check_regeneration(self, sim):
        if self.is_regeneration_state():    #is system empty?
            current_time = sim.current_time #get current time
            if current_time > self.cycle_start_time:    #check whether we are not still at the same moment
                T = sim.current_time - self.cycle_start_time    #determine cycle time
                self.cycle_time.append(T)   #add cycle time to list
                self.cycle_waiting_hist.append(self.cycle_waiting_sum.copy())   #add the waiting time of this cycle to the list
                self.cycle_queue_hist.append(self.cycle_queue_area.copy())  #add the queue length form thsi cycle to the list
                self.cycle_busy_hist.append(self.cycle_busy_area.copy())    #add the busy time of the trucks of the cycle to the list
                self.cycle_arrivals_hist.append(self.cycle_arrivals.copy())  #add the number of arrivals from the cycle to the list
                self.cycle_completed_hist.append(self.cycle_completed.copy())

                #reset
                self.cycle_start_time = current_time
                self.cycle_waiting_sum = [0.0]*self.N
                self.cycle_queue_area = [0.0] * self.N
                self.cycle_busy_area = [0.0] * self.M
                self.cycle_completed = [0]*self.N
                self.cycle_arrivals = [0]*self.N

class Arrival(Event):
    def __init__(self, time, district, model):
        super().__init__(time)
        self.district = district
        self.model = model

    def execute(self, sim):
        model = self.model
        model.update_time_stats(sim)

        #Add request to queue
        waste_type = random.choices([1,2,3], weights = [model.q1, model.q2, model.q3])[0]   #initilize wate type
        request = {                                 #initialize request
            "arrival_time": sim.current_time,
            "type": waste_type,
            "district": self.district
        }
        model.queues[self.district].append(request) #put request in queue
        model.arrivals[self.district] +=1   #add an arrival
        model.cycle_arrivals[self.district] += 1 #add a arrival to this cycle

        #Next arrival
        rate = model.lambdas[self.district] #get the rate
        next_time = sim.current_time + random.expovariate(rate) #determine next arrival with an poisson process
        sim.schedule(Arrival(next_time, self.district, model))  #schedule next arrival

        #Try to start service?
        model.try_dispatch(sim) #check idle trucks

        for j, truck in enumerate(model.trucks):    #check for rerouting
            if truck["home"] != self.district:  #check only the hom truck
                continue
            if truck["status"] == "busy" and truck["location"] != truck["home"]:    #if truck is busy and not at home
                home = truck["home"]    #determine home of truck
                if len(model.queues[home]) > model.K:   #if the length of the queue is bigger then the threshold
                    sim.schedule(Reroute(sim.current_time, j, home, model)) #schedule rerouting

class ServiceCompletion(Event):
    def __init__(self, time, truck_id, district, model):
        super().__init__(time)
        self.truck_id = truck_id
        self.district = district
        self.model = model

    def execute(self, sim):
        model = self.model
        model.update_time_stats(sim)

        truck = model.trucks[self.truck_id] #get truck id
        request = truck["current_request"]  #get request that truck is currently wokring on

        waiting_time = sim.current_time - request["arrival_time"]   #determine the total waiting time
        model.total_waiting_time[request["district"]] += waiting_time   #add waiting time to list of waiting times per district
        model.cycle_waiting_sum[request["district"]]+= waiting_time #add waiting time to list of waiting times per district for this cycle
        model.cycle_completed[request["district"]] +=1 #increase completed counter for this cycle
        model.completed +=1 #increase general completed counter

        #update truck statuses
        truck["status"] = "idle"    #set truck to idle
        truck["current_request"] = None #remove request
        truck["completion_event"] = None    #remove completion event

        #trigger truck decision
        model.truck_decision(self.truck_id, sim)

class Reroute(Event):
    def __init__(self, time, truck_id, district, model):
        super().__init__(time)
        self.truck_id = truck_id
        self.district = district
        self.model = model

    def execute(self, sim):
        model = self.model
        model.update_time_stats(sim)

        truck = model.trucks[self.truck_id] #get truck id

        #cancel current service
        if truck["completion_event"]:   #if the truck had an completion event
            sim.cancel(truck["completion_event"])   #cancel the completion event
        request = truck["current_request"]  #get the request the truck is currently working on
        model.queues[truck["location"]].insert(0, request)  #put the request back in the queue of its original district in the front
        truck["status"] = "idle"    #set truck to idle
        truck["current_request"] = None #remove current request
        truck["completion_event"] = None    #remove completion event

        #Increase counter
        model.reroutings +=1    #increase number of reroutings

        #start with the service at home
        model.truck_decision(self.truck_id, sim)

def confidence_interval(X, T):  #determine the CI
    X = np.array(X, dtype=float)    #get the queue length/waiting time/busy time per cycle
    T = np.array(T, dtype=float)    #get the total time/total arrives per cycle
    N = len(X) #determine number of cycles
    X_mean = np.mean(X) #means
    T_mean = np.mean(T)
    theta = X_mean / T_mean #determine desired value
    var_X = np.sum((X - X_mean) ** 2) / (N - 1) #variances
    var_T = np.sum((T - T_mean) ** 2) / (N - 1)
    cov_XT = np.sum((X - X_mean)*(T-T_mean)) / (N-1)    #covariance
    var_V = var_X + theta ** 2 * var_T - 2 * theta * cov_XT #variance
    se = np.sqrt(var_V / (T_mean ** 2 * N)) #standard error
    t_value = t.ppf(1 - 0.05 / 2, df=N - 1) #critical value
    half_width = t_value * se   #determine half width
    lower = theta - half_width  # detemine lower ci
    upper = theta + half_width  # determine upper ci
    return theta, lower, upper

if __name__ == "__main__":
    sim = Simulation()
    model = CityWasteModel()

    for district in range(model.N):
        sim.schedule(Arrival(0.0, district, model))

    sim.on_after_event(lambda s, e: model.check_regeneration(s))

    sim.run(lambda s: len(model.cycle_time) >= 200) #todo: bedenk ff hoe dat zit met iteraties

    print("\n=== Expected Steady-State Waiting Time ===")
    for district in range(model.N):
        X = [] #total waiting time per cycle
        T = [] #number of completed jobs per cycle
        for cycle in range(len(model.cycle_waiting_hist)):
            X.append(model.cycle_waiting_hist[cycle][district])
            T.append(model.cycle_completed_hist[cycle][district])
        theta, lower, upper = confidence_interval(X, T)
        print(f"District {district}: "f"{theta:.4f} "f"(95% CI: [{lower:.4f}, {upper:.4f}])")

    # Queue length per district
    print("\n=== Expected Steady-State Queue Length ===")
    for district in range(model.N):
        X =[]
        for cycle in model.cycle_queue_hist:
            X.append(cycle[district])
        theta, lower, upper = confidence_interval(X, model.cycle_time)
        print(f"District {district}: "f"{theta:.4f} "f"(95% CI: [{lower:.4f}, {upper:.4f}])")

    # Utilisation per truck
    print("\n=== Truck utilisation per truck ===")
    for truck in range(model.M):
        X = []
        for cycle in model.cycle_busy_hist:
            X.append(cycle[truck])
        theta, lower, upper = confidence_interval(X, model.cycle_time)
        print(f"Truck {truck}: "f"{theta:.4f} "f"(95% CI: [{lower:.4f}, {upper:.4f}])")

    print("Rerouting rate:", model.reroutings / sim.current_time)












