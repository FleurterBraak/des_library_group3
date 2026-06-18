import random
from core import *
import numpy as np

#Time helpers
def hour(t):    #determine hour of a day
    return t % 24

def weekday(t): #determine day of the week (0=Mon,...,6=Sun)
    return int(t // 24) % 7

def is_office_hour(t):  #determines whether we are in office hours or not
    if weekday(t) < 5:  #if the day is monday to friday
        if 8 <= hour(t) < 16:   #if we are between 8 and 6
            return True #return true
        else:
            return False    #night
    else:
        return False    #weekend

def next_change_of_office_state(t): #determines the next time we change between office hours and outside office hours (used for splitting utilization)
    day_start = int(t // 24) * 24   #compute the start of the current day
    candidates = [] #create list to store possible times when office hours change
    candidates.append(day_start + 8)    #today at 8
    candidates.append(day_start + 16)   #today at 16
    candidates.append(day_start + 24 + 8)   #tomorrow at 8
    candidates.append(day_start + 24 + 16)  #today at 8
    actual_candidates = []  #create a list for moments that are after t
    for candi in candidates:    #loop through al the moments
        if candi > t:   #if they are after t
            actual_candidates.append(candi) #append them to the list
    next_change = min(actual_candidates)    #take the minimum as next change
    return next_change  #return


# Inpatient arrival rate
def inpatient_rate(t):  # determining inpatient rate at a certain time
    lambda_I = 3/8  # base rate
    h = hour(t)     # get hour of the day
    if weekday(t) >= 5:  # in the weekend there are no office hours, so we only have the base rate
        return lambda_I
    if 9 <= h <= 15:    # hour between 09:00 and 15:00
        return lambda_I + 3 * (np.sin(np.pi / 3 * (h - 9))**2)  #function for the rate of lambda
    else:
        return lambda_I #outside 9 and 15 we only have the base rate
    #return 0

def thinning_next(t, max_rate=3.375):   #computing the next arrival time of an inpatient, note 3.375<=lambda_I
    while True: #we keep generating arrival times until one is accepted
        t += random.expovariate(max_rate)   #generate an exponential interarrival time and add it to t
        if random.random() < inpatient_rate(t) / max_rate:  #determine ratio with current rate, then randomly choose whether it is accepted or not. So, if the true rate is high, we accept with high probability
            return t    #return the time of next arrival



class CTModel:
    def __init__(self):
        # Rates
        self.lambda_E = 0.01   #arrival rate emergency patients (24 per day = 1 per hour)
        self.lambda_O = 23/8    #arrival rate outpatients (during office hours)

        # Waiting room
        self.waiting_room_capacity = 3  #number of chairs in waiting room
        self.FIFO_queue = []    #queue of waiting patients
        self.seats_taken = 0    #number of seats in waiting room that is occupied

        # Scanners
        self.scanner = [False, False]   #booleans for a scanner is busy (true) of idle (false)

        # Patients/scheduling
        self.last_event_time = 0.0  # time of last event
        self.calendar = {}  #calendar for outpatient scheduling, maps (day, hour) to number outpatients with appointment
        self.waiting_list = []  #waiting list of outpatients over the weekend
        self.inpatient_in_waiting_room = False  #whether an inpatient is already waiting
        self.inpatient_in_transport = False #whether an inpatient is in transport
        self.pending_inpatient_requests = 0 #number of inpatients that are still waiting in another part of the hospital

        # Batch statistics
        self.warmup_time = 0.0  #the time the warmup takes
        self.batch_start_time = None    #moment when batch starts
        self.min_batches = 20   #minimum number of batches
        self.max_batches = 2000 #maximum number of batches
        self.target_precision = 0.05   #5% relative precision
        self.batch_length = 0   #length of a batch
        self.current_batch = 0  #witch batch we are currently at

        self.batch_emergency_wait_sum = 0.0 #total waiting time of all emergency patients in a batch
        self.batch_emergency_wait_count = 0 #number of emergency patients in a batch
        self.batch_outpatient_wait_sum = 0.0    #total waiting time of all outpatients in a batch
        self.batch_outpatient_wait_count = 0    #number of outpatients in a batch
        self.batch_patients = 0 #total number of patients arriving in a batch
        self.batch_patients_outside = 0 #number of patients that have to wait outsie
        self.batch_inpatients_office_hours = 0  #number of inpatients that arrive during office hours
        self.batch_inpatients_not_same_day = 0  #number of inpatients that arrive during office hours and cannot be scanned the same day
        self.batch_utilization_area = 0.0   #time scanners are busy during a batch
        self.batch_utilization_office_hours = 0.0   #time scanners are busy during office hours during a batch
        self.batch_utilization_outside_office_hours = 0.0   #time scanners are busy outside office hours during a batch
        self.batch_capacity_area = 0.0  #time scanners capacity during a batch
        self.batch_capacity_office_hours = 0.0  #time scanners capacity during office hours during a batch
        self.batch_capacity_outside_office_hours = 0.0  #time scanners capacity outside office hours during a batch
        self.batch_access_times = 0.0   #total time between a call and appointment for outpatients in a batch
        self.batch_access_counter = 0.0 #number of schedulled outpatients

        self.batch_emergency_wait_history = []  #list to store waiting times emergency patients
        self.batch_outpatient_wait_history = [] #list to store waiting times outpatients
        self.batch_outside_history = [] #list to store fraction of patients that have to wait outside
        self.batch_inpatient_history = []   #list to store fraction of inpatients that cannot be scanned same day
        self.batch_utilization_history = [] #list to store utilization
        self.batch_utilization_office_hours_history = []    #list to store utilization during office hours
        self.batch_utilization_outside_office_hours_history = []    #list to store utilization outside office hours
        self.batch_access_history = []  #list to store access times for outpatients

    #Time updating (utilization)
    def update_time(self, sim): #function called at each event
        t0 = self.last_event_time   #set t0 to last event
        t1 = sim.current_time   #set t1 to current event to create an interval
        self.last_event_time = t1   #update last_event_time
        busy = sum(self.scanner)    #determine the number of busy scanners
        t = t0  #set t to t0
        while t < t1:   #loop through the interval
            next_boundary = min(t1, next_change_of_office_state(t)) #set next boundry to the min of the end of the interval or the next change in office hours
            dt = next_boundary - t  #determine time inbetween
            capacity = self.scanner_capacity(t) #determine capacity

            if sim.current_time >= self.warmup_time:    #if warmup is completed
                self.batch_utilization_area += busy * dt    #add the busy time to the utilization
                self.batch_capacity_area += capacity * dt   #add the capacity time to the batch capacity
                if is_office_hour(t):   #when in office hours
                    self.batch_utilization_office_hours += busy * dt    #add the busy time to the utilization for office hours
                    self.batch_capacity_office_hours += capacity * dt   #add the capacity time to the batch capactity for office hhours
                else:   #outisde office hours
                    self.batch_utilization_outside_office_hours += busy * dt    #add the busy time to the utilization outside office hours
                    self.batch_capacity_outside_office_hours += capacity * dt   #add the capacity time to the batch capacity outside office hours

            t = next_boundary   #set t to next boundary

    #Waiting room and queue
    def enter_waiting_room(self, patient, sim):
        self.FIFO_queue.append(patient) #add patient to the queue
        if sim.current_time >= self.warmup_time:    #if warmup is completed
            self.batch_patients += 1    #increase counter of arrived patients
        if self.seats_taken < self.waiting_room_capacity:   #if there is still a seat available in waiting room
            self.seats_taken += 1   #set that seat to occupied
            patient["inside"] = True    #set patients inside to true
        else:   #all seats occupied
            patient["inside"] = False  #set patients inside to false
            if sim.current_time >= self.warmup_time:    #if warmup completed
                self.batch_patients_outside += 1    #increase counter of patients that have to wait outside

    def get_next_patient(self): #determine which patient we get from the queue
        if not self.FIFO_queue: #if queue is empty
            return None #no patient can be scanned
        for patient in self.FIFO_queue: #loop through all patients
            if patient["type"] == "E":  #if there is an emergency patients
                self.FIFO_queue.remove(patient) #remove that patient from queue
                if patient.get("inside", False):    #check if that patient was inside the waiting room
                    self.seats_taken -= 1   #if so, set seat to free
                return patient  #return the emergency patient to be scanned
        patient = self.FIFO_queue.pop(0)    #else get first patient from the queue
        if patient.get("inside", False):    #check if that patient was inside the waiting room
            self.seats_taken -= 1   #if so, set seat to free
        return patient  #return that patient to be scanned

    #Scanner capacity and scan start
    def scanner_capacity(self, t):  #determine ow many scanner are available at time t
        if is_office_hour(t):   #check if t is during office hours
            return 2    #return 2
        else:   #if t is outside office hours
            return 1    #return 1
        #return 2

    def try_start_scan(self, sim):
        capacity = self.scanner_capacity(sim.current_time)  #get capacity
        for scanner in range(capacity): #loop through the scanners that are on
            if not self.scanner[scanner]:   #if scanner is idle
                patient = self.get_next_patient()   #get next patient from queue
                if not patient: #if there is not patient
                    return  #return
                self.scanner[scanner] = True    #set scanner to busy
                patient["start"] = sim.current_time #set start time of patient to the current time

                if patient["type"] == "E":  #if patient is an emergency patient
                    wait = patient["start"] - patient["arrival"]    #calculate waiting time
                    if sim.current_time >= self.warmup_time:    #if warmup is completed
                        self.batch_emergency_wait_sum += wait   #add waiting time to the total waiting time of emergency patients
                        self.batch_emergency_wait_count += 1    #increase counter

                elif patient["type"] == "O":    #if patient is an outpatients
                    wait = patient["start"] - patient["arrival"]    #calculate waiting time
                    if sim.current_time >= self.warmup_time:    #if warmup is completed
                        self.batch_outpatient_wait_sum += wait  #add waiting time to the total waiting time of outpatients
                        self.batch_outpatient_wait_count += 1   #increase counter

                elif patient["type"] == "I":    #if patient is an inpatient
                    request_time = patient["request_time"]  #get request time
                    if patient.get("office_hour_request", False):   #if patient arrived during office hours
                        request_day = int(request_time // 24)   #determine request day
                        start_day = int(patient["start"] // 24) #determine day scan starts
                        same_day_before_16 = (request_day == start_day and hour(patient["start"]) < 16) #determine whether the scan started the same day within office hours
                        if not same_day_before_16:  #if that is not the case
                            if sim.current_time >= self.warmup_time:    #if warmup is completed
                                self.batch_inpatients_not_same_day += 1 #increase counter of inpatients not scanned on the same day
                    model.pending_inpatient_requests -= 1   #decrease counter
                    model.inpatient_in_waiting_room = False
                    if model.pending_inpatient_requests > 0: #if there are still inpatient requests
                        model.inpatient_in_transport = True #set next inpatient in transport
                        next_arrival = sim.current_time + random.uniform(9/60, 15/60)   #determine next arrival time
                        sim.schedule(InpatientArrival(next_arrival, model, patient["request_time"]))    #schedule next arrival

                sim.schedule(ScanCompletion(sim.current_time + patient["service"], patient, scanner, self)) #schedule the scan completion

    #Outpatient scheduling
    def slot_capacity(self, h): #determine slot capacity
        if 8 <= h < 12: #if it is morning
            return 4    #return 4
        elif 12 <= h < 16:  #if it is afternoon
            return 3    #return 3
        else:   #else
            return 0    #return 0

    def schedule_outpatient(self, sim):
        current_time = sim.current_time #get current time
        current_day = int(current_time // 24)   #determine current day
        current_weekday = weekday(current_time) #determine current weekday
        start_day = current_day + 1 #determine the first day outpatient can be scheduled (1 day after call)
        days_until_friday = max(0, 4 - current_weekday) #determine how many days there are until friday
        friday = current_day + days_until_friday    #determine which day is the next friday

        for day in range(start_day, friday + 1):    #loop through all days from the start day to friday
            if weekday(day * 24) >= 5:  #skip the weekend (just to make sure)
                continue
            for h in range(8, 16):  #loop over the office hours
                key = (day, h)  #make a calendar key
                if self.calendar.get(key, 0) < self.slot_capacity(h):   #check if slot still has capacity
                    self.calendar[key] = self.calendar.get(key, 0) + 1  #add patient to that slot
                    appointment_time = day * 24 + h #determine simulation time appointment
                    access_time_days = (appointment_time - current_time) / 24   #calculate access time
                    if sim.current_time >= self.warmup_time:    #if warmup is completed
                        self.batch_access_times += access_time_days #add the access time to the total access times
                        self.batch_access_counter += 1  #increase counter of scheduled outpatients
                    sim.schedule(OutpatientArrival(appointment_time, self)) #schedule arrival of outpatient
                    return
        self.waiting_list.append({"request_time": current_time})    #if there is no slot available, place outpatient on waiting list

    def flush_waiting_list(self, sim): #calles every friday at four
        if not self.waiting_list:   #check if waiting list is empty
            return  #if so, return
        current_day = int(sim.current_time // 24)   #detemrine current day
        current_weekday = weekday(sim.current_time) #determine current weekday
        days_until_monday = (7 - current_weekday)   #determine days until monday
        next_week_monday = current_day + days_until_monday  #determine the next monday
        new_waiting_list = []   #create new waiting list

        for request in self.waiting_list:   #loop over all requests on the waiting list
            request_time = request["request_time"]  #get request time
            scheduled = False   #set request to not scheduled
            for day in range(next_week_monday, next_week_monday + 5):   #loop through all days next week
                if weekday(day * 24) >= 5:  #make sure to skip weekends
                    continue
                for h in range(8, 16):  #loop through the office hours
                    key = (day, h)  #create calendar key
                    if self.calendar.get(key, 0) < self.slot_capacity(h):   #check if the is still capacity
                        self.calendar[key] = self.calendar.get(key, 0) + 1  #if so, schedule appointment
                        appointment_time = day * 24 + h #determine appointment time in simulation
                        sim.schedule(OutpatientArrival(appointment_time, self)) #schedule arrival of outpatient
                        access_time_days = (appointment_time - request_time) / 24   #determine access time
                        if sim.current_time >= self.warmup_time:    #if warmup is completed
                            self.batch_access_times += access_time_days #add access time to total access times
                            self.batch_access_counter += 1  #increase counter of schedules outpatients
                        scheduled = True    #set request is scheduled to true
                        break   #break
                if scheduled:   #if request is schedule (just to verify if we have request that are scheduled earlier)
                    break   #break
            if not scheduled:   #if not scheduled
                new_waiting_list.append(request)    #place is on the waiting list for next week
        self.waiting_list = new_waiting_list    #set waiting list to the new one

    #Batch end
    def batch_end(self, sim):   #called after each event
        if sim.current_time < self.warmup_time: #if we are still in warmup
            return  #skip
        if self.batch_start_time is None:   #if there is no batch start time (first batch)
            self.batch_start_time = sim.current_time    #set batch start time to current time
        if sim.current_time >= self.batch_start_time + self.batch_length:   #if batch is ended
            print(f"Batch {self.current_batch + 1} finished")   #debug print statement
            if self.batch_emergency_wait_count > 0: #if there arrived an emergency patient during the batch
                self.batch_emergency_wait_history.append(self.batch_emergency_wait_sum / self.batch_emergency_wait_count)   #append the average waiting time to the history
            else:
                self.batch_emergency_wait_history.append(0.0)   #append 0
            if self.batch_outpatient_wait_count > 0:    #if there arrive an outpatient during the batch
                self.batch_outpatient_wait_history.append(self.batch_outpatient_wait_sum / self.batch_outpatient_wait_count)    #append the average waiting time to the history
            else:
                self.batch_outpatient_wait_history.append(0.0)  #append 0
            if self.batch_patients > 0: #if there arrive patients during the batch
                self.batch_outside_history.append(self.batch_patients_outside / self.batch_patients)    #add fraction of patients that had to wait outside the waiting room to the history
            else:
                self.batch_outside_history.append(0.0)  #append 0
            if self.batch_inpatients_office_hours > 0:  #if inpatients arrived during office hours
                self.batch_inpatient_history.append(self.batch_inpatients_not_same_day / self.batch_inpatients_office_hours)    #add fraction of inpatients that could not be scanned the same day to the history
            else:
                self.batch_inpatient_history.append(0.0)    #append 0
            self.batch_utilization_history.append(self.batch_utilization_area / self.batch_capacity_area)   #add utilization to history
            self.batch_utilization_office_hours_history.append(self.batch_utilization_office_hours / self.batch_capacity_office_hours)  #add utilization during office horus to history
            self.batch_utilization_outside_office_hours_history.append(self.batch_utilization_outside_office_hours / self.batch_capacity_outside_office_hours)  #add utilization outside office hours to history
            if self.batch_access_counter > 0:   #if an outpatient scheduled an appointment
                self.batch_access_history.append(self.batch_access_times / self.batch_access_counter)   #add average access time to history
            else:
                self.batch_access_history.append(0.0)   #append 0

            # reset batch counters
            self.batch_emergency_wait_sum = 0.0
            self.batch_emergency_wait_count = 0
            self.batch_outpatient_wait_sum = 0.0
            self.batch_outpatient_wait_count = 0
            self.batch_patients = 0
            self.batch_patients_outside = 0
            self.batch_inpatients_office_hours = 0
            self.batch_inpatients_not_same_day = 0
            self.batch_utilization_area = 0.0
            self.batch_utilization_office_hours = 0.0
            self.batch_utilization_outside_office_hours = 0.0
            self.batch_capacity_area = 0.0
            self.batch_capacity_office_hours = 0.0
            self.batch_capacity_outside_office_hours = 0.0
            self.batch_access_times = 0.0
            self.batch_access_counter = 0.0

            self.batch_start_time += self.batch_length  #update batch start time
            self.current_batch += 1 #increase current batch

            if self.current_batch % 10 == 0:    #debug print stament
                print(f"Current batch = {self.current_batch}")
                report_precisions(self)

#Events
class ScanCompletion(Event):    #is called when a scan is completed
    def __init__(self, time, patient, scanner, model):
        super().__init__(time)
        self.patient = patient
        self.scanner = scanner
        self.model = model

    def execute(self, sim):
        model = self.model
        model.update_time(sim)  #update time
        model.scanner[self.scanner] = False #set scanner to idle
        model.try_start_scan(sim)   #try to start scan

class EmergencyArrival(Event):  #is called when an emergency patient arrives
    def __init__(self, time, model):
        super().__init__(time)
        self.model = model

    def execute(self, sim):
        model = self.model
        model.update_time(sim)  #update time
        model.enter_waiting_room({  #let patient enter waiting room
            "type": "E",    #type emergency
            "arrival": sim.current_time,    #arrival time is current time
            "service": random.uniform(10/60, 19/60) #service time is random uniform with 10 to 19 min
        }, sim)
        model.try_start_scan(sim)   #try to start scan
        if model.lambda_E > 0:  #if arrival rate is bigger than zero
            sim.schedule(EmergencyArrival(  #schedule a new arrival of an emergency patient
                sim.current_time + random.expovariate(model.lambda_E),
                model
            ))

class OutpatientRequest(Event): #is called when an outpatient calls for an appointment
    def __init__(self, time, model):
        super().__init__(time)
        self.model = model

    def next_arrival_time(self, t):
        if self.model.lambda_O == 0:    #if rate is 0, then do nothing (verification case)
            return None
        while True: #loop until we have a next arrival
            t += random.expovariate(self.model.lambda_O)    #determine possible next arrival
            if weekday(t) < 5 and 8 <= hour(t) < 16:    #check if it is office hours
                return t    #return time of next arrival

    def execute(self, sim):
        model = self.model
        model.update_time(sim)  #update time
        model.schedule_outpatient(sim)  #schedule outpatient
        next_time = self.next_arrival_time(sim.current_time)    #determine next arrival time
        if next_time is not None:   #if that exist
            sim.schedule(OutpatientRequest(next_time, model))   #schdedule next arrival

class WeeklyFlush(Event):   #called every friday at four
    def __init__(self, time, model):
        super().__init__(time)
        self.model = model

    def execute(self, sim):
        model = self.model
        model.update_time(sim)  #update time
        model.flush_waiting_list(sim)   #call flush_waiting list

        # Schedule next Friday 16:00
        t = sim.current_time    #get current time
        day = int(t // 24)  #determine day
        day += 1    #go to next day
        while weekday(day * 24) != 4:  # 4 = Friday, loop until friday is found
            day += 1    #go to next day
        next_friday_16 = day * 24 + 16  #determine simultion time next friday at four
        sim.schedule(WeeklyFlush(next_friday_16, model))    #schedule next flush

class InpatientRequest(Event):  #when a request to scan an inpatient arrives
    def __init__(self, time, model):
        super().__init__(time)
        self.model = model

    def execute(self, sim):
        model = self.model
        model.update_time(sim)  #update time
        request_time = sim.current_time #get request time (that is now)
        model.pending_inpatient_requests += 1   #increase counter
        if not model.inpatient_in_waiting_room and not model.inpatient_in_transport:    #if no inpatient is waiting or in transport
            model.inpatient_in_transport = True #put patient in transport
            arrival_time = request_time + random.uniform(9/60, 15/60)   #determine arrival time (inpatients have travel time)
            sim.schedule(InpatientArrival(arrival_time, model, request_time))   #schedule arrival

        if inpatient_rate(sim.current_time) > 0:    #if inpatient rate is bigger than 0 (verification case)
            next_time = thinning_next(request_time) #determine next request time
        else:
            next_time = None    #otherwise no request
        if next_time is not None:   #if there is a new request time
            sim.schedule(InpatientRequest(next_time, model))    #schedule new arrival of request

class InpatientArrival(Event):
    def __init__(self, time, model, request_time):
        super().__init__(time)
        self.model = model
        self.request_time = request_time

    def execute(self, sim):
        model = self.model
        model.update_time(sim)  #update time
        model.inpatient_in_transport = False    #remove inpatient from transport
        model.inpatient_in_waiting_room = True  #set inpatient in waitingroom
        arrival_time = sim.current_time #get arrival time
        request_time = self.request_time    #get request time
        office_hour_request = is_office_hour(request_time)  #check if the request is in office hours

        #define the patient
        patient = {
            "type": "I",
            "arrival": arrival_time,
            "request_time": request_time,
            "service": random.uniform(10/60, 19/60),
            "office_hour_request": office_hour_request
        }

        if office_hour_request: #if the request came during office hours
            if sim.current_time >= model.warmup_time:  #if warmup completed
                model.batch_inpatients_office_hours += 1    #increase counter of inpatients arriving during office hours

        model.enter_waiting_room(patient, sim)  #trigger enter waiting room
        model.try_start_scan(sim)   #trigger try start scan

class OutpatientArrival(Event): #when an outpatient arrives in the CT department
    def __init__(self, time, model):
        super().__init__(time)
        self.model = model

    def execute(self, sim):
        model = self.model
        model.update_time(sim)  #update time
        if random.random() < 0.84:  #outpatient only show up with probability 0.84
            model.enter_waiting_room({  #enter waiting room
                "type": "O",    #type outpatient
                "arrival": sim.current_time,    #arrival time
                "service": random.uniform(10/60, 19/60) #service time
            }, sim)
        model.try_start_scan(sim)   #trigger try start scan

#Steady-state statistics
def confidence_interval(batch_values):
    X = np.array(batch_values, dtype=float)
    n = len(X)
    mean = np.mean(X)   #calculate mean
    std = np.std(X, ddof=1) #calculate standard deviation
    z = 1.96 #set z value (95%)
    half_width = z * std / np.sqrt(n)   #determine half width
    lower = mean - half_width   #determine lower CI
    upper = mean + half_width   #determine upper CI
    return mean, lower, upper


def check_precision(mean, lower, upper, delta=0.05):
    h = (upper - lower) / 2 #get the halfwidth of the CI
    if mean == 0:   #mean should not be zero
        return False, np.inf
    relative_precision = h / mean   #compute relative precision
    output = (relative_precision <= delta)  #check whether that forfills the tollerance
    return output, relative_precision

def precision_satisfied(model, delta=0.05):
    metrics = [
        model.batch_emergency_wait_history,
        model.batch_outpatient_wait_history,
        model.batch_outside_history,
        model.batch_inpatient_history,
        model.batch_utilization_history,
        model.batch_utilization_office_hours_history,
        model.batch_utilization_outside_office_hours_history,
        model.batch_access_history,
    ]
    for values in metrics:
        if len(values) < 2: #we want at least 2 batches
            return False
        mean, lower, upper = confidence_interval(values)    #get 95% CI
        is_satisfied, _ = check_precision(mean, lower, upper, delta)    #check if condition is satisfied
        if not is_satisfied:    #if not return False
            return False
    return True

#Warm-up determination
def determine_warmup_time(N_W, K_W, epsilon):   #this is described in lecture 2 of StochSim
    runs = [] #make a list to store the runs
    for n in range(N_W):    #for N_W simulation runs (small number), run the simulation
        sim = Simulation()
        model = CTModel()
        waiting_times = []   #store waiting times
        times = []  #sotre start times

        # Monkey-patch try_start_scan to record ALL waiting times
        old_try = CTModel.try_start_scan    #safe original try_start_scan so that for now we can replace it

        def new_try_start_scan(self, sim):
            capacity = self.scanner_capacity(sim.current_time)  #get capacity
            for scanner in range(capacity): #loop through working scanners
                if not self.scanner[scanner]:   #if scanner is idle
                    patient = self.get_next_patient()   #get next patient from queue
                    if not patient: #if there is none
                        return  #return
                    self.scanner[scanner] = True    #set scanner to busy
                    patient["start"] = sim.current_time #set start time of patient
                    wait = patient["start"] - patient["arrival"]    #determine waiting time
                    waiting_times.append(wait)  #append waiting time to list
                    times.append(patient["start"])  #Append start times to list

                    if len(waiting_times) % 1000 == 0:  #debug print statement
                        print(f"Warmup run: {len(waiting_times)} waiting times collected")

                    #schedule scan completion
                    sim.schedule(ScanCompletion(
                        sim.current_time + patient["service"],
                        patient,
                        scanner,
                        self
                    ))

        CTModel.try_start_scan = new_try_start_scan #replace try start scan with this new one

        # Schedule events
        if model.lambda_E > 0:  #if lambda_E is bigger than 0
            sim.schedule(EmergencyArrival(0, model))    #schedule an emergency arrival
        if model.lambda_O > 0:  #if lambda_O is bigger than 0
            sim.schedule(OutpatientRequest(0, model))   #schedule an outpatient request
        if inpatient_rate(0) > 0:   #if lambda_I is bigger than 0
            sim.schedule(InpatientRequest(0, model))    #schedule an inpatient request
        sim.schedule(WeeklyFlush(4 * 24 + 16, model))   #schedule weekly flusch

        sim.run(lambda s: len(waiting_times) >= K_W)    #run until K_W waiting times collected
        runs.append((waiting_times[:K_W], times[:K_W])) #append to runs
        CTModel.try_start_scan = old_try    #set original function back

    # Compute W_bar_k (mean across runs)
    W = np.array([r[0] for r in runs])  #create a matrix of the runs
    T = np.array([r[1] for r in runs])

    W_bar_k = np.mean(W, axis=0) #point estimator for the expected waiting time of the k-th run
    T_bar_k = np.mean(T, axis=0)

    for D in range(100, K_W // 2): #find a D such that it is at least 100 and 2D<K_W
        mean_D = np.mean(W_bar_k[:D])   #calculate mean with D (slides)
        mean_2D = np.mean(W_bar_k[:2 * D])  #calculate mean with 2D (slides)
        ratio = abs(mean_2D / mean_D - 1)   #get the absolute reatio
        if ratio <= epsilon:    #check whether it is smaller then the tollerence
            return T_bar_k[D]  # warm-up time
    return T_bar_k[K_W // 2]    #in case criterion is not met at all


#Helper function for debugging
def report_precisions(model):
    names = [   #name all metrics
        "Emergency wait",
        "Outpatient wait",
        "Outside fraction",
        "Inpatient not same day",
        "Utilisation overall",
        "Utilisation office hours",
        "Utilisation outside office hours",
        "Access time"]
    metrics = [ #get al metrics
        model.batch_emergency_wait_history,
        model.batch_outpatient_wait_history,
        model.batch_outside_history,
        model.batch_inpatient_history,
        model.batch_utilization_history,
        model.batch_utilization_office_hours_history,
        model.batch_utilization_outside_office_hours_history,
        model.batch_access_history,]
    print("\nRelative precisions:") #print all metrics with their precision
    for name, values in zip(names, metrics):
        mean, lower, upper = confidence_interval(values)
        _, rp = check_precision(mean, lower, upper, model.target_precision)
        print(f"{name:35s} {rp:.4f}")



if __name__ == "__main__":
    print("Determining warm-up period:")
    warmup_time = determine_warmup_time(N_W=1, K_W=20000, epsilon=0.05)
    print("Estimated warm-up:", warmup_time)

    sim = Simulation()
    model = CTModel()

    model.warmup_time = warmup_time
    model.batch_length = 4 * warmup_time  # rule-of-thumb batch length

    print("Batch length:", model.batch_length)

    #if arrival rate is postive, schedule an arrival/request
    if model.lambda_E > 0:
        sim.schedule(EmergencyArrival(0, model))
    if model.lambda_O > 0:
        sim.schedule(OutpatientRequest(0, model))
    if inpatient_rate(0) > 0:
        sim.schedule(InpatientRequest(0, model))

    first_friday_16 = 4 * 24 + 16   #determine first friday
    sim.schedule(WeeklyFlush(first_friday_16, model))   #schdule first weekly flush

    sim.on_after_event(lambda s, e: model.batch_end(s))

    #run until precision is met or max number batches
    sim.run(lambda s: (model.current_batch >= model.min_batches and precision_satisfied(model, model.target_precision)) or model.current_batch >= model.max_batches)

    print(f"\nNumber of batches used: {model.current_batch}")
    report_precisions(model)

    print("\n=== Emergency Waiting Time ===")
    theta, lower, upper = confidence_interval(model.batch_emergency_wait_history)
    print(theta, f"95% CI:[{lower}, {upper}]")

    print("\n=== Outpatient Waiting Time ===")
    theta, lower, upper = confidence_interval(model.batch_outpatient_wait_history)
    print(theta, f"95% CI:[{lower}, {upper}]")

    print("\n=== Fraction Waiting Outside ===")
    theta, lower, upper = confidence_interval(model.batch_outside_history)
    print(theta, f"95% CI:[{lower}, {upper}]")

    print("\n=== Inpatient Not Same Day (request in office hours) ===")
    theta, lower, upper = confidence_interval(model.batch_inpatient_history)
    print(theta, f"95% CI:[{lower}, {upper}]")

    print("\n=== CT Utilization (overall) ===")
    theta, lower, upper = confidence_interval(model.batch_utilization_history)
    print(theta, f"95% CI:[{lower}, {upper}]")

    print("\n=== CT Utilization (office hours) ===")
    theta, lower, upper = confidence_interval(model.batch_utilization_office_hours_history)
    print(theta, f"95% CI:[{lower}, {upper}]")

    print("\n=== CT Utilization (outside office hours) ===")
    theta, lower, upper = confidence_interval(model.batch_utilization_outside_office_hours_history)
    print(theta, f"95% CI:[{lower}, {upper}]")

    print("\n=== Outpatient access time (days) ===")
    theta, lower, upper = confidence_interval(model.batch_access_history)
    print(theta, f"95% CI:[{lower}, {upper}]")
