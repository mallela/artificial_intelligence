import random
# Simple reflex agent : returns action for the current percept
def simpleReflexAgent(latestPercept):
	# status1 = status of current room. 
	# status2 = status of next room.
	location,status1, status2 = latestPercept
	if status1 == 'dirty': 
		r1= 'suck'
	elif location=='A': 
		r1='right'
	elif location =='B':
		r1= 'left'
	if status2 == 'dirty': 
		r2= 'suck'
	elif location=='A': 
		r2='left'
	elif location =='B':
		r2= 'right'
	return r1,r2

if __name__ == '__main__':
	lifetime=1000 
	# Initializing a dictionary for tracking individual performance measure
	# each element is of the format '(location1, status1, status2)'.
	# Only current location is mentioned
	performanceMeasure = {('A','clean','dirty'):0,('A','dirty','dirty'):0, 
							('A','clean','clean'):0,('A','dirty','clean'):0,
							('B','clean','dirty'):0,('B','dirty','dirty'):0, 
							('B','clean','clean'):0,('B','dirty','clean'):0}

	# All sets of (current location 1, status of room 1, status of second room)
	perceptSequence = [('A','clean','dirty'),('A','dirty','dirty'), 
							('A','clean','clean'),('A','dirty','clean'),
							('B','clean','dirty'),('B','dirty','dirty'), 
							('B','clean','clean'),('B','dirty','clean')]
	for i in range(1,lifetime):
		# choosing a random percept
		randomConfiguration = random.choice(perceptSequence)
		# Calling a simple reflex agent to deduce action
		action1, action2 = simpleReflexAgent(randomConfiguration)
		print "Action for ", randomConfiguration , " is ", action1
		if action1 == 'suck' : 
			performanceMeasure[randomConfiguration]=performanceMeasure[randomConfiguration]+1
		if action2 == 'suck':
			performanceMeasure[randomConfiguration]=performanceMeasure[randomConfiguration]+1
		action = (action1,action2)
	print "\nThe different configurations and their respective performance measures\nof the agent over a life time of ",lifetime," steps are :\n",performanceMeasure
	print "\nThe total performance measure is ", sum(performanceMeasure.values())
	#  average performance over a lifetime
	print "\nThe average performance over a", lifetime, "steps is", float(sum(performanceMeasure.values()))/lifetime