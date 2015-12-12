from __future__ import division
from math import log10
import numpy as np
import json
import sys

how_to = "Please give: \n" \
		 "--run noweight --dumpfactor <float number> --numtasks <total tasks> --lastcompleted <position of last completed task>" \
		 " \nOR\n" \
		 "--run weighted1 || weighted2 --dumpfactor <float number> --numtasks <total tasks> --completed <taskNumber1> = <value1>, <taskNumber2> = <value2>" \
		 " \nOR\n" \
		 "--run weighted3 --dumpfactor <float number> --numtasks <total tasks> --completed <taskNumber1> = <value1>, <taskNumber2> = <value2> --time <taskNumber1> = <value1>, <taskNumber2> = <value2>"


## Consider:
# 1. completion rate of task itself and even distribution of out-edges-weight among rest of tasks
# 1. completion rates of other tasks
# 2. time expiration rates of other tasks
def WeightedSCRank3(dumping_factor, num_tasks, completion_rates, time_rates):

	sum = 0
	## Find the total remaining time for all the tasks
	for time in time_rates:
		sum += time_rates[time]

	## For each task compute the **IMPORTANCE** of time
	for time in time_rates:
		time_rates[time] = abs(log10(time_rates[time]/sum))


	Vprev = np.array([1/num_tasks]*num_tasks)
	Tax = np.array([(1-dumping_factor)/num_tasks]*num_tasks)

	rows = []
	for task in range(0,num_tasks):
		## node 0 is the value 0+1 inside the quotas dictionary
		## every outgoing edge should have the value of the completed tasks multiplied by
		same_node_weight_final = 1-completion_rates[task+1]
		other_node_weight_final = completion_rates[task+1]/(num_tasks-1)

		sum = 0
		row = []
		##
		for curr_pos in range(0,num_tasks):
			if curr_pos != task:
				other_node_weight = other_node_weight_final * (1-completion_rates[curr_pos+1]) * time_rates[curr_pos+1]
				row.append(other_node_weight)
				sum += other_node_weight
			else:
				same_node_weight = same_node_weight_final * (1-completion_rates[curr_pos+1]) * time_rates[curr_pos+1]
				row.append(same_node_weight)
				sum += same_node_weight
		row[:] = [x/sum for x in row]
		rows.append(row)


	Transition_matrix = np.array(rows)
	Transition_matrixT = np.transpose(Transition_matrix)

	error=float('inf')
	num_iters = 0
	threshold = 0.1
	Vnext = None
	while (error > threshold):
		Vnext = np.dot(dumping_factor*Transition_matrixT,Vprev) + Tax
		error = np.linalg.norm((Vnext-Vprev))
		Vprev = Vnext
		num_iters += 1


	print("Taxation: ", Tax)
	print("Transition Matrix Transpose: ", Transition_matrixT)
	print("Vector Computed: ", Vnext)
	print(np.sum(Vnext))
	print("Error: ", error)
	print("Iterations Performed: ", num_iters)

	return Vnext



## Consider:
# 1. completion rate of task itself and even distribution of out-edges-weight among rest of tasks
# 2. completion rates of other tasks
def WeightedSCRank2(dumping_factor, num_tasks, completion_rates):

	# dumping_factor = 1
	# num_tasks = 3
	# completion_rates = {1:0.1,2:0.5,3:0.8}

	Vprev = np.array([1/num_tasks]*num_tasks)
	Tax = np.array([(1-dumping_factor)/num_tasks]*num_tasks)

	rows = []
	for task in range(0,num_tasks):
		## node 0 is the value 0+1 inside the quotas dictionary
		## every outgoing edge should have the value of the completed tasks multiplied by
		same_node_weight_final = 1-completion_rates[task+1]
		other_node_weight_final = completion_rates[task+1]/(num_tasks-1)

		sum = 0
		row = []
		for curr_pos in range(0,num_tasks):
			if curr_pos != task:
				other_node_weight = other_node_weight_final * (1-completion_rates[curr_pos+1])
				row.append(other_node_weight)
				sum += other_node_weight
			else:
				same_node_weight = same_node_weight_final * (1-completion_rates[curr_pos+1])
				row.append(same_node_weight)
				sum += same_node_weight
		row[:] = [x/sum for x in row]
		rows.append(row)


	Transition_matrix = np.array(rows)
	Transition_matrixT = np.transpose(Transition_matrix)

	error=float('inf')
	num_iters = 0
	threshold = 0.1
	Vnext = ""
	while (error > threshold):
		Vnext = np.dot(dumping_factor*Transition_matrixT,Vprev) + Tax
		error = np.linalg.norm((Vnext-Vprev))
		Vprev = Vnext
		num_iters += 1

	print("Taxation: ", Tax)
	print("Transition Matrix Transpose: ", Transition_matrixT)
	print("Vector Computed: ", Vnext)
	print(np.sum(Vnext))
	print("Error: ", error)
	print("Iterations Performed: ", num_iters)

	return Vnext

## Consider:
# 1. completion rate of task itself and even distribution of out-edges-weight among rest of tasks
def WeightedSCRank1(dumping_factor, num_tasks, completion_rates):

	# dumping_factor = 1
	# num_tasks = 10
	# completion_rates = {1:0.05,2:0.5,3:0.2,4:0.5,5:0.9,6:0.7,7:0.8,8:0.9,9:0.1,10:0.95}


	Vprev = np.array([1/num_tasks]*num_tasks)
	Tax = np.array([(1-dumping_factor)/num_tasks]*num_tasks)
	rows = []
	for task in range(0,num_tasks):
		## node 0 is the value 0+1 inside the quotas dictionary
		## every outgoing edge should have the value of the completed tasks multiplied by
		same_node_weight = 1-completion_rates[task+1]
		other_node_weight = completion_rates[task+1]/(num_tasks-1)
		row = [other_node_weight if curr_pos != task else same_node_weight for curr_pos in range(0,num_tasks)]
		rows.append(row)

	Transition_matrix = np.array(rows)
	Transition_matrixT = np.transpose(Transition_matrix)

	error=float('inf')
	num_iters = 0
	threshold = 0.1
	Vnext = ""
	while (error > threshold):
		Vnext = np.dot(dumping_factor*Transition_matrixT,Vprev) + Tax
		error = np.linalg.norm((Vnext-Vprev))
		Vprev = Vnext
		num_iters += 1

	print("Taxation: ", Tax)
	print("Transition Matrix Transpose: ", Transition_matrixT)
	print("Vector Computed: ", Vnext)
	print(np.sum(Vnext))
	print("Error: ", error)
	print("Iterations Performed: ", num_iters)

	return Vnext


# Consider:
# 1. Last completed task -- No incoming edge for this task from the rest of the nodes
def NoWeightSCRank(dumping_factor, num_tasks, last_completed_task):
	# dumping_factor = 0.85
	Vprev = np.array([1/num_tasks]*num_tasks)
	Tax = np.array([(1-dumping_factor)/num_tasks]*num_tasks)
	rows = []
	position = last_completed_task-1
	for task in range(0,num_tasks):
		if task != position:
			## include all the tasks; exclude the last completed
			row = [1/(num_tasks-1) if curr_pos!=position else 0 for curr_pos in range(0,num_tasks)]
		else:
			row = [1/num_tasks for curr_pos in range(0,num_tasks)]
		rows.append(row)

	Transition_matrix = np.array(rows)
	Transition_matrixT = np.transpose(Transition_matrix)

	error=float('inf')
	num_iters = 0
	threshold = 0.1
	Vnext = ""
	while (error > threshold):
		Vnext = np.dot(dumping_factor*Transition_matrixT,Vprev) + Tax
		error = np.linalg.norm((Vnext-Vprev))
		Vprev = Vnext
		num_iters += 1


	print("Taxation: ", Tax)
	print("Transition Matrix Transpose: ", Transition_matrixT)
	print("Vector Computed: ", Vnext)
	print(np.sum(Vnext))
	print("Error: ", error)
	print("Iterations Performed: ", num_iters)

	return Vnext



def CallNoWeightSCRank(sys_args):
	if len(sys_args)>4 and sys_args[3] == "--dumpfactor":
		dumping_factor = float(sys_args[4])
		if len(sys_args)>6 and sys_args[5] == "--numtasks":
			num_tasks = int(sys_args[6])
			if len(sys_args)>7 and sys_args[7] == "--lastcompleted":
				last_completed = int(sys_args[8])
				NoWeightSCRank(dumping_factor, num_tasks, last_completed)
			else:
				print("Plase give the number of the last completed task")
		else:
			print("Please give the number of tasks")
	else:
		print("Please give the dumping factor")


def CallWeightedSCRank12(sys_args, alg_name):
	if len(sys_args)>4 and sys_args[3] == "--dumpfactor":
		dumping_factor = float(sys_args[4])
		if len(sys_args)>6 and sys_args[5] == "--numtasks":
			num_tasks = int(sys_args[6])
			if len(sys_args)>7 and sys_args[7] == "--completed":
				i=8
				completion_rates = {}
				while i<len(sys_args):
					task = (sys_args[i].split("="))
					completion_rates[int(task[0])] = float(task[1])
					i+=1
				if alg_name == "weighted1":
					print WeightedSCRank1(dumping_factor, num_tasks, completion_rates)
				elif alg_name == "weighted2":
					print WeightedSCRank2(dumping_factor, num_tasks, completion_rates)
			else:
				print("Please give the completed rates of the tasks")
	else:
		print("Please give the total number of the tasks")


def CallWeightedSCRank3(sys_args):
	if len(sys_args)>4 and sys_args[3] == "--dumpfactor":
		dumping_factor = float(sys_args[4])
		if len(sys_args)>6 and sys_args[5] == "--numtasks":
			num_tasks = int(sys_args[6])
			if len(sys_args)>7 and sys_args[7] == "--completed":
				i=8
				completion_rates = {}
				while i<len(sys_args) and sys_args[i] != "--time":
					task = (sys_args[i].split("="))
					completion_rates[int(task[0])] = float(task[1])
					i+=1
				if i==len(sys_args) or len(sys_args)<i+2:
					print("Please give the remaining time of the tasks in Minutes OR in the same time unit metric(e.g. seconds, hour)")
				else:
					i+=1
					time_rates = {}
					while i<len(sys_args):
						task = (sys_args[i].split("="))
						time_rates[int(task[0])] = float(task[1])
						i += 1
					print(WeightedSCRank3(dumping_factor, num_tasks, completion_rates, time_rates))
			else:
				print("Please give the completed rates of the tasks")
	else:
		print("Please give the total number of the tasks")


if __name__ == "__main__":

	if len(sys.argv)>1 and sys.argv[1] == "--run":
			run_mode = sys.argv[2]
			if run_mode == "noweight":
				CallNoWeightSCRank(sys.argv)
			elif run_mode == "weighted1":
				CallWeightedSCRank12(sys.argv, "weighted1")
			elif run_mode == "weighted2":
				CallWeightedSCRank12(sys.argv, "weighted2")
			elif run_mode == "weighted3":
				CallWeightedSCRank3(sys.argv)
	else:
		print(how_to)
