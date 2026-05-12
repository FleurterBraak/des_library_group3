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

        self.warmup_time = 5000 #set warmup length todo: decide how long it should be (met die delta ofzo)
        self.batch_length = 2000    #set batch length   todo: decide how long
        self.number_batches = 30    #set number of batches  todo: how many
        self.current_batch = 0  #tracker of at which batch we are
        self.batch_start_time = self.warmup_time    #set the start time of batches to the end of the warmup time

        self.batch_waiting = [0.0]*N    #current batch waiting times
        self.batch_completed = [0]*N    #current number of completed request of this batch
        self.batch_queue_area = [0.0]*N #current queue length over time in this badge
        self.batch_busy_area = [0.0]*M  #current time truck is busy in the batch

        self.batch_waiting_hist = []    #list for all waiting times from all batches
        self.batch_completed_hist = []  #list for number of completed request from all batches
        self.batch_queue_hist = []  #list for all queue lengths of all batches
        self.batch_busy_hist = []   #list for all busy times of trucks of all batches

    def update_time_stats(self, sim):   #called at every event to update the times
        dt = sim.current_time - self.last_event_time    #time interval between two events
        self.last_event_time = sim.current_time #update last event time

        if sim.current_time < self.warmup_time:
            return

        for district in range(self.N):  #compute queue lengths
            q = len(self.queues[district])  #all request waiting
            for truck in self.trucks:   #all request beining served
                if truck["status"] == "busy" and truck["location"] == district:
                    q += 1
            self.batch_queue_area[district] += q * dt   #update the queuelength of this batch

        for j, truck in enumerate(self.trucks): #compute utility of truck
            if truck["status"] == "busy":   #for all trucks that are busy update
                self.batch_busy_area[j] += dt   #update time truck was busy for this batch

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

    def batch_end(self, sim):
        if sim.current_time >= self.batch_start_time + self.batch_length:   #if end of batch reached
            self.batch_waiting_hist.append(self.batch_waiting.copy())   #put all variables in history list
            self.batch_completed_hist.append(self.batch_completed.copy())
            self.batch_queue_hist.append(self.batch_queue_area.copy())
            self.batch_busy_hist.append(self.batch_busy_area.copy())

            #reset current counters
            self.batch_waiting = [0.0]*self.N
            self.batch_completed = [0]*self.N
            self.batch_queue_area = [0.0]*self.N
            self.batch_busy_area = [0.0]*self.M
            self.batch_start_time += self.batch_length
            self.current_batch += 1

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
        if sim.current_time >= model.warmup_time:
            model.batch_waiting[request["district"]] += waiting_time
            model.batch_completed[request["district"]] += 1
        model.completed +=1 #increase general completed counter todo: moet dit hier wel staan?

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
        if sim.current_time >= model.warmup_time:
            model.reroutings +=1    #increase number of reroutings

        #start with the service at home
        model.truck_decision(self.truck_id, sim)

def confidence_interval(batch_values):  #determine the CI
    X = np.array(batch_values, dtype=float)
    n = len(X)
    mean = np.mean(X)   #calculate mean
    std = np.std(X, ddof=1) #calculate standard deviation
    t_value = t.ppf(1 - 0.05 / 2, df=n - 1) #calculate critical value
    half_width = t_value * std / np.sqrt(n) #determine half width
    lower = mean - half_width  # determine lower ci
    upper = mean + half_width  # determine upper ci
    relative_precision = half_width / mean  #todo: iets doen met deze check moet onder 0.10 zijn
    print(relative_precision)
    return mean, lower, upper

if __name__ == "__main__":
    sim = Simulation()
    model = CityWasteModel()

    for district in range(model.N):
        sim.schedule(Arrival(0.0, district, model))

    sim.on_after_event(lambda s, e: model.batch_end(s))

    sim.run(lambda s: model.current_batch >= model.number_batches)

    print("\n=== Expected Steady-State Waiting Time ===")
    for district in range(model.N):
        batch_estimates = []

        for batch in range(model.number_batches):
            completed = model.batch_completed_hist[batch][district]
            if completed > 0:
                total_wait = model.batch_waiting_hist[batch][district]
                batch_estimates.append(total_wait / completed)
        theta, lower, upper = confidence_interval(batch_estimates)
        print(f"District {district}: " f"{theta:.4f} " f"(95% CI: [{lower:.4f}, {upper:.4f}])")

    # Queue length per district
    print("\n=== Expected Steady-State Queue Length ===")
    for district in range(model.N):
        batch_estimates = []

        for batch in range(model.number_batches):
            area = model.batch_queue_hist[batch][district]
            batch_estimates.append(area / model.batch_length)
        theta, lower, upper = confidence_interval(batch_estimates)
        print(f"District {district}: " f"{theta:.4f} " f"(95% CI: [{lower:.4f}, {upper:.4f}])")

    # Utilisation per truck
    print("\n=== Truck utilisation per truck ===")
    for truck in range(model.M):
        batch_estimates = []
        for batch in range(model.number_batches):
            busy = model.batch_busy_hist[batch][truck]
            batch_estimates.append(busy / model.batch_length)
        theta, lower, upper = confidence_interval(batch_estimates)
        print(f"Truck {truck}: " f"{theta:.4f} " f"(95% CI: [{lower:.4f}, {upper:.4f}])")

    print("Rerouting rate:", model.reroutings / (sim.current_time - model.warmup_time))