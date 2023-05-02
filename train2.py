import random

#train weights for use in text generation

#file = str(open('raye_responses.txt', 'r', encoding = 'utf-8').read()).split('\n')
#file2 = str(open('training_data.txt', 'r', encoding = 'utf-8').read()).split('\n')
#for i in file2:
#	file.append(i)
	
file = str(open('training_data_4.txt', 'r', encoding = 'utf-8').read()).split('\n')
#print(file)


#first dict is frequency of each word
wordFreq = {}
wordFreq['SOL'] = 0
wordFreq['EOL'] = 0
for i in file:
	words = ['SOL'] + i.split(' ') + ['EOL']
	for j in words:
		if j in wordFreq:
			wordFreq[j] += 1
		else:
			wordFreq[j] = 1

#second dict is frequency of each pair of words
#a pair of words exists in the sentence if the second word appears within three indicies after the first word
pairFreq = {}
for i in file:
	words = ['SOL'] + i.split(' ') + ['EOL']
	for j in range(len(words)):
		curr = words[j]
		for k in range(1,4):
			if j < len(words) - k:
				next = words[j+k]
				if str(str(curr) + " " + str(next)) in pairFreq:
					pairFreq[str(str(curr) + " " + str(next))] += 1
				else:
					pairFreq[str(str(curr) + " " + str(next))] = 1




#3rd dict is probability of each pair
#first, minimum nonzero probability will be calculated
#then temp values will be put into pairFreq and pairProbability
pairProbability = {}
for key, val in pairFreq.items():
	firstWord = key.split(' ')[0]
	secondWord = key.split(' ')[1]
	probability = pairFreq[key] / wordFreq[firstWord]
	pairProbability[key] = probability

bias = (pairProbability[min(pairProbability, key=pairProbability.get)]) / float(len(wordFreq))
#print(minNonzero)

#still on the second dict, populate it with combinations of words that dont exist (but technically could) 
#give them a freq of 0
#for key, val in wordFreq.items():
#	firstWord = key
#	for key2, val2 in wordFreq.items():
#		newPair = str(str(firstWord) + " " + str(key2))
#		if newPair not in pairFreq:
#			pairFreq[newPair] = 0
#			pairProbability[newPair] = 0.0

#print(len(pairProbability))

#4th dict is weights
pairWeights = {}
for key, val in pairFreq.items():
	pairWeights[key] = 1.0



#dicts:
#wordFreq - frequency of each word
#pairFreq - frequency of each pair of words (size of wordFreq squared)
#pairProbability - probability of each pair of words
#pairWeight - weight for each, will start out at 1.

#variables:
#bias - the minimum nonzero probability, which will be added to weighted probabilities later

#now that these have been ititialized, the algorithm can start.
#classify a sentence as either making sense or not making sense 
#if a sentence makes sense, the sum of probabilities * weights / sum of unweighted probabilities >= 1
#if a sentence is classified as making sense, but the user inputs that it should not make sense
	#the weights of that sentence are adjusted to reflect that some of its probabilities are misleading
	#decrease all weights.
#if a sentence is classified as not making sense, but the user inputs that it should make sense
	#the weights indicate that the sentence should not make sense, which means they are too low
	#increase all weights.
#if a sentence is classified correctly, the weights are correct. Nothing needs to be done.


file = str(open('training_data5.txt', 'r').read()).split('\n')


iterations = 200

for iter in range(iterations):

	numCorrect = 0
	numIncorrect = 0
	for i in file:
		sense = int(i[0])
		inp = i[1:len(i)]
		
		#classify sentence:
		#first, get tuples from input
		inp = ['SOL'] + inp.split(' ') + ['EOL']
		
		inputTuples = []
		
		for i in range(len(inp)):
			curr = inp[i]
			
			for j in range(1,4):
				if i < len(inp) - j:
					next = inp[i+j]
					inputTuples.append(str(str(curr) + " " + str(next)))
				
			
		#inputTuples is initialized
		weightedSum = 0.0
		unweightedSum = 0.0
		for i in inputTuples:
			try:
				x = pairWeights[i]
			except:
				pairWeights[i] = 1.0
			try:
				x = pairProbability[i]
			except:
				pairProbability[i] = 0.0
			weightedSum = weightedSum + (pairWeights[i] * pairProbability[i] + bias)
			unweightedSum = unweightedSum + (pairProbability[i] + bias)
		
		result = weightedSum / unweightedSum
		
		
		#adjust weights accordingly
		if result >= 1 and sense == 0:
			numIncorrect += 1
			for i in inputTuples:
				pairWeights[i] = pairWeights[i] * 0.999
		elif result < 1 and sense == 1:
			numIncorrect += 1
			for i in inputTuples:
				pairWeights[i] = pairWeights[i] * 1.001
		else:
			numCorrect += 1

	accuracy = float(numCorrect) / (numCorrect + numIncorrect)
	print("accuracy: " + str(accuracy))

#write results to a file
newFile = open("claire_word_pair_weights.txt", 'w')
for k, v in pairWeights.items():
	newFile.write(str(k) + ' ' + str(v) + '\n')






	
