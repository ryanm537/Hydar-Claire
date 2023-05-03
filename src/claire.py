import random
import time
import subprocess

#first
#take input
#get a context map for that input, and keywords
#search questions/answers files for two matching questions
#One question will match the context map of the input. Save the context map of that question's answer
#The second question will match the keywords of the input. Find the keywords of its answer and use those as parameters for generate text.
#the generated text have a similar context map to the first question's answer, to an extent.



#generate
file = str(open('claire_data/claire_training_data_4.txt', 'r').read()).split('\n')
#print(file)

#cleans input
#removes punctuation, changes plurals to singulars if possible, and changes 'nt to two words
#this is not necessarily always desirable but it makes it easier to find parts of speech
def clean(inp):
	p = ",./?!;:\"'><()@#$%^&*-=_+"
	inp = inp.split(' ')
	cleanedInp = []
	for i in inp:
		i = str(i)		
		#first, remove any punctuation
		for j in i:
			if j in p:
				i = i.replace(j, "")				
		#second, replace plurals	
		if len(i)-1>=0 and i[len(i)-1] == 's' and i not in partsOfSpeech and i[0:len(i)-1] in partsOfSpeech:
			i = i[0:len(i)-1]
		#third, replace "nt"
		if len(i)-2 >= 0 and i[len(i)-2:len(i)] == 'nt' and i not in partsOfSpeech and i[0:len(i)-2] in partsOfSpeech:
			i = i[0:len(i)-2]
			cleanedInp.append(i.lower())
			cleanedInp.append("not")
		else:
			cleanedInp.append(i.lower())
	result = ""
	for i in cleanedInp:
		result = result + ' ' + i
	result = result[1:len(result)]
	return result

#returns "U" for words that dont have parts of speech in the dict
def lookupPos(word):
	try:
		pos = partsOfSpeech[word]
	except:
		pos = 'U'
	return pos

#parts of speech dict for fast lookup of words
f = str(open("claire_data/claire_parts_of_speech.txt", "r").read()).split('\n')
partsOfSpeech = {}
for i in f:
	i = i.split('#')
	if len(i[1]) > 1:
		i[1] = i[1][0:1] + '+'
	partsOfSpeech[i[0]] = i[1]

#print("parts of speech dict initialized")

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


#print("word frequency dict initialized")

#second dict is frequency of each pair of words
pairFreq = {}
for i in file:
	words = ['SOL'] + i.split(' ') + ['EOL']
	for j in range(len(words) - 1):
		curr = words[j]
		next = words[j+1]
		
		if str(str(curr) + " " + str(next)) in pairFreq:
			pairFreq[str(str(curr) + " " + str(next))] += 1
		else:
			pairFreq[str(str(curr) + " " + str(next))] = 1


#print("word pair frequency dicts initialized")


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



#print("word pair probability dicts initialized")
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

#print("zeroes added to word pair freq dicts")
#4th dict is weights
pairWeights = {}
#for key, val in pairFreq.items():
#	pairWeights[key] = 1.0

weightFile = open("claire_data/claire_word_pair_weights.txt", 'r').read().split('\n')
for line in weightFile:
	line = line.split(' ')
	try:
		k = line[0] + ' ' + line[1]
		v = line[2]
		pairWeights[k] = float(v)
	except:
		pass
#for key, value in pairProbability.items():
#	print((key, value))

#print("pair weights dict initialized")
#dicts:
#wordFreq - frequency of each word
#pairFreq - frequency of each pair of words (size of wordFreq squared)
#pairProbability - probability of each pair of words
#pairWeight - weight for each, will start out at 1.

#variables:
#bias - the minimum nonzero probability, which will be added to weighted probabilities later



#check if sentence makes sense
# first check will be using word pair method which uses the existing weights.
# will need to create a new feature and probability dict for this check

pair_feature_freq = {}
for i in file:
	words = ['SOL'] + i.split(' ') + ['EOL']
	for j in range(len(words)):
		curr = words[j]
		for k in range(1,4):
			if j < len(words) - k:
				next = words[j+k]
				if str(str(curr) + " " + str(next)) in pair_feature_freq:
					pair_feature_freq[str(str(curr) + " " + str(next))] += 1
				else:
					pair_feature_freq[str(str(curr) + " " + str(next))] = 1

pair_feature_probability = {}
for key, val in pair_feature_freq.items():
	firstWord = key.split(' ')[0]
	secondWord = key.split(' ')[1]
	probability = pair_feature_freq[key] / wordFreq[firstWord]
	pair_feature_probability[key] = probability

pair_feature_bias = (pair_feature_probability[min(pair_feature_probability, key=pair_feature_probability.get)]) / float(len(wordFreq))
#print(minNonzero)


#populate probabilities and freqs with combinations of words that dont exist (but technically could) 
#give them a freq of 0
#for key, val in wordFreq.items():
#	firstWord = key
#	for key2, val2 in wordFreq.items():
#		newPair = str(str(firstWord) + " " + str(key2))
#		if newPair not in pair_feature_freq:
#			pair_feature_freq[newPair] = 0
#			pair_feature_probability[newPair] = 0.0


#print("word pair features dicts initialized")
#second sense check is for parts of speech.
#all dicts for word freq, weight, probability, and pair freq will need to be initialized for pos.


#initialize frequencies of each unique part of speech
posFreq = {}
posFreq[lookupPos('SOL')] = 0
posFreq[lookupPos('EOL')] = 0

for i in file:
	i = ['SOL'] + clean(i).split(' ') + ['EOL']
	for j in i:
		pOfS = lookupPos(j)
		if pOfS not in posFreq:
			posFreq[pOfS] = 1
		else:
			posFreq[pOfS] += 1;
	

#initialize frequencies of each feature
featureFreq = {}
for i in file:
	i = ['SOL'] + clean(i).split(' ') + ['EOL']
	for j in range(len(i)):
		firstPofS = lookupPos(i[j])
		for k in range(1,4):
			if j + k < len(i):
				secondPOfS = lookupPos(i[j+k])
				feature = str(k) + ' ' + firstPofS + ' ' + secondPOfS
				if feature not in featureFreq:
					featureFreq[feature] = 1
				else:
					featureFreq[feature] += 1

#initialize probability of a feature occurring
#number of times that the second part of speech ocurrs at the indicated distance from the first part of speech
	#divided by the total number of times the first part of speech ocurrs.
P_feature = {}
for k, v in featureFreq.items():
	firstPOfS = k.split(' ')[1]
	P = float(v) / float(posFreq[firstPOfS])
	P_feature[k] = P


part_of_speech_bias = (P_feature[min(P_feature, key=P_feature.get)]) / float(len(posFreq))

#add zero probabilities (and frequencies) for features that do not exist in the data but technically could
for k1, v1 in posFreq.items():
	for k2, v2 in posFreq.items():	
		for k in range(1, 4):
			possibleFeature = str(k) + ' ' + k1 + ' ' + k2
			if possibleFeature not in featureFreq:
				featureFreq[possibleFeature] = 0
				P_feature[possibleFeature] = 0.0


#load weights dict from file
part_of_speech_weights = {}
posWeightFile = open("claire_data/claire_part_of_speech_weights.txt", 'r').read().split('\n')

for line in posWeightFile:
	line = line.split(' ')
	try:
		k = line[0] + ' ' + line[1]
		v = line[2]
		part_of_speech_weights[k] = float(v)
	except:
		pass


#print("part of speech feature dicts initialized")

#initilaize dicts for context mapping


#context_map_feature_freq - frequency of each feature
context_map_feature_freq = {}
#relevantFreq - frequency of which a feature was found to be relevant
relevantFreq = {}
#irrelevantFreq - frequency of which a feature was found to be irrelevant
irrelevantFreq = {}
#P_relevant - probability of which a feature was relevant
P_relevant = {}
#P_irrelevant - probability of which a feature was irrelevant
P_irrelevant = {}

#open data2
training_data2 = str(open("claire_data/claire_training_data2.txt", "r").read()).split('\n\n')
for line in training_data2:
	contents = line.split('\n')
	#parse the sentence from a line in data2
	sentence = clean(contents[0])
	#parse the true context map given for that line in data2
	givenMap = contents[1:len(contents)]
	reformattedGivenMap = []
	for i in givenMap: #i now looks like "[7, 8],['lazy', 'dog']"
		i = i.split('], [') # i now looks like ["[7, 8"], ["'lazy', 'dog']"]
		indicies = i[0][1:len(i[0])].split(',') # indicies looks like ['7', '8']
		givenPair = i[1][0:len(i[1])-1].replace("'", "").split(', ') # givenPair looks like ["lazy", "dog"]
		dist = abs(int(indicies[1]) - int(indicies[0]))
		reformattedPair = str(dist) + ' ' + lookupPos(givenPair[0]) + ' ' + lookupPos(givenPair[1])
		reformattedGivenMap.append(reformattedPair)
	
	#get all possible context features in that line
	sentence = ['SOL'] + sentence.split(' ') + ['EOL']
	for i in range(len(sentence)):
		currentPartOS = lookupPos(sentence[i])
		for j in range(1, 3):
			if i + j < len(sentence):
				nextPartOS = lookupPos(sentence[i+j])
				currentFeature = str(j) + ' ' + currentPartOS + ' ' + nextPartOS
				#add that feature to holistic feature frequencies map
				if currentFeature not in posFreq:
					context_map_feature_freq[currentFeature] = 1
				else:
					context_map_feature_freq[currentFeature] += 1
				
				#add that feature to either relevant or irrelevant features
				#depending on whether it is found in reformattedGivenMap
				if currentFeature not in reformattedGivenMap:
					if currentFeature not in irrelevantFreq:
						irrelevantFreq[currentFeature] = 1
					else:
						irrelevantFreq[currentFeature] += 1
				else:
					if currentFeature not in relevantFreq:
						relevantFreq[currentFeature] = 1
					else:
						relevantFreq[currentFeature] += 1

#now that posFreq, relevantFreq, and irrelevantFreq are initialized, calculate probabilities
for k, v in context_map_feature_freq.items():
	if k in relevantFreq:
		P_relevant[k] = float(relevantFreq[k]) / float(v)
	if k in irrelevantFreq:
		P_irrelevant[k] = float(irrelevantFreq[k]) / float(v)

#biasR = (P_relevant[min(P_relevant, key=P_relevant.get)]) / float(len(context_map_feature_freq))
#biasIR = (P_irrelevant[min(P_irrelevant, key=P_irrelevant.get)]) / float(len(context_map_feature_freq))

#print("context mapping dicts initialized")

#open question and response files
questions = open("claire_data/claire_questions.txt", 'r').read().split('\n')
answersFile = open("claire_data/claire_responses.txt", 'r')
answers = {}
#initialize answers as a dict
index = 0
for line in answersFile:
	line = line.replace('\n', '')
	answers[index] = line
	index += 1

#print("answer file dict initialized")

#add features that could exist but arent found in the data
#give the probabilities and freqs of 0
for k, v in posFreq.items():
	firstPOS = k
	for k1, v1 in posFreq.items():
		for i in range(1, 3):
			secondPOS = k1
			pair = str(i) + ' ' + firstPOS + ' ' + secondPOS
			if pair not in context_map_feature_freq:
				context_map_feature_freq[pair] = 0
				relevantFreq[pair] = 0
				irrelevantFreq[pair] = 0
				P_relevant[pair] = 0.0
				P_irrelevant[pair] = 0.0
			if pair not in relevantFreq:
				P_relevant[pair] = 0.0
				relevantFreq[pair] = 0
			if pair not in irrelevantFreq:
				P_irrelevant[pair] = 0.0
				irrelevantFreq[pair] = 0

#print("all dicts initialized")

#create two values - context map and keyword list
#training_data2 = str(open("training_data2.txt", "r").read()).split('\n\n')
def contextMap(input):
	#open data2
	#training_data2 = str(open("training_data2.txt", "r").read()).split('\n\n')

	sentence = input

	createdContextMap = []
	createdWordMap = []
	#get all possible context features in that line
	sentence = sentence.split(' ')
	for i in range(len(sentence)):
		currentPartOS = lookupPos(sentence[i])
		for j in range(1, 3):
			if i + j < len(sentence):
				nextPartOS = lookupPos(sentence[i+j])
				currentFeature = str(j) + ' ' + currentPartOS + ' ' + nextPartOS
				#add that feature to created context map if it is found to be relevant
				#use probabilities to determine relevance
				try:
					isRelevant = (P_relevant[currentFeature])
					isIrrelevant = P_irrelevant[currentFeature]
					prediction = isRelevant - isIrrelevant
					if prediction >= 0:
						#print(prediction)
						createdContextMap.append(currentFeature)
						createdWordMap.append([(str(i) + ' ' + str(i+j)), str(sentence[i]) + ' ' + str(sentence[i+j])])
				except:
					doNothing = 0

	#extract keywords
	keywordFreqs = {}
	for i in createdWordMap:
		i = i[0].split(' ')
		for j in i:
			if j in keywordFreqs:
				keywordFreqs[j] += 1
			else:
				keywordFreqs[j] = 1
	
	top1 = (0, 0)
	top2 = (0, 0)
	iS = input.split(' ')
	for k, v in keywordFreqs.items():
		if v > top1[1] and (lookupPos(iS[int(k)])[0] == 'N' or lookupPos(iS[int(k)])[0] == 'V' or lookupPos(iS[int(k)])[0] == 't' or lookupPos(iS[int(k)])[0] == 'i'):
			top2 = top1
			top1 = (k, v)
		if v > top2[1] and k != top1[0] and (lookupPos(iS[int(k)])[0] == 'N' or lookupPos(iS[int(k)])[0] == 'V' or lookupPos(iS[int(k)])[0] == 't' or lookupPos(iS[int(k)])[0] == 'i'):
			top2 = (k, v)
	keywords = input.split(' ')[int(top1[0])] + " " + input.split(' ')[int(top2[0])]
	return [keywords, createdContextMap]



#given a starting word, find a random second word by using a weighted probability
#weighted probability is the weight of that pair times the probability of that pair plus the bias

#generate - generate a sentence using a centered word (string) parameter, called keyword
#returns a string

#targetMap is a context map
#pairs of words that match targetMap will be given more priority
def generate(keyword, targetMap):
	input = "the"
	#file = str(open('training_data.txt', 'r').read()).split('\n')
	input = keyword
	#first, extract all pairs that begin with the input from pairProbabilities to obtain those probabilities
	#append those probabilities to a list and the corresponding word to a second list
	next = [input]
	prev = [input]
	nextHalf = []
	firstHalf = []
	solPriority = 1.0
	eolPriority = 1.0
	while prev[0] != 'SOL':
		firstHalf = prev + firstHalf
		probabilities = []
		samples = []
		for key, val in wordFreq.items():
			sample = str(key) + ' ' + prev[0]
			try:
				tryWeight = pairWeights[sample]
			except:
				#there is possibly a key error there
				#add that key to weights, probabilities, and frequencies
				pairWeights[sample] = 1.0
			try:
				tryWeight = pairProbability[sample]
			except:
				pairProbability[sample] = 0.0
				pairFreq = 0
			
			#give increased priority to SOL to end sentences faster
			probability = pairWeights[sample] * (pairProbability[sample] + bias)
			if str(key) == 'SOL':
				probability = probability * solPriority
			
			#give increased priority if adding sample to the sentence creates a target context feature
			for i in range(1, 4):
				if (i-1) < len(firstHalf):
					sampleContextFeature = str(i) + ' ' + lookupPos(clean(str(key))) + ' ' + lookupPos(clean(firstHalf[i-1]))
					if sampleContextFeature in targetMap:
						probability = probability * 8.0
			
			#print(probability)
			samples.append(str(key))
			probabilities.append(probability)
		prev[0] = random.choices(samples, weights = probabilities, k=1)[0]
		if prev[0] != 'SOL':
			#increase solPriority exponentially over time
			solPriority = solPriority * 4.0
		if '.' in prev[0] or '?' in prev[0] or '!' in prev[0]: #or ',' in prev[0]
			break
		#print(next)
	#print(firstHalf)
	
	while next[0] != 'EOL':
		nextHalf = nextHalf + next
		probabilities = []
		samples = []
		for key, val in wordFreq.items():
			sample = next[0] + " " + str(key)
			try:
				tryWeight = pairWeights[sample]
			except:
				#there is possibly a key error there
				#add that key to weights, probabilities, and frequencies
				pairWeights[sample] = 1.0
			try:
				tryWeight = pairProbability[sample]
			except:
				pairProbability[sample] = 0.0
				pairFreq = 0
			
			probability = 0.0
			if str(key) == 'EOL':
				probability = pairWeights[sample] * (pairProbability[sample] + bias) * eolPriority
			else:
				probability = pairWeights[sample] * (pairProbability[sample] + bias)
			
			#give increased priority if adding sample to the sentence creates a target context feature
			for i in range(1, 4):
				if len(nextHalf)-i >= 0:
					sampleContextFeature = str(i) + ' ' + lookupPos(clean(nextHalf[len(nextHalf)-i])) + ' ' + lookupPos(clean(str(key)))
					if sampleContextFeature in targetMap:
						probability = probability * 8.0
			
			#print(probability)
			samples.append(str(key))
			probabilities.append(probability)
		next[0] = random.choices(samples, weights = probabilities, k=1)[0]
		if next[0] != 'EOL':
			#increase solPriority exponentially over time
			eolPriority = eolPriority * 4.0
		#print(next)
	
	sentence = firstHalf + nextHalf[1:len(nextHalf)]
	sentenceStr = ""
	for i in sentence:
		sentenceStr = sentenceStr + ' ' + i
	sentenceStr = sentenceStr[1:len(sentenceStr)]
	return (sentenceStr)




#sense_check_word_pair
# takes a string as input which represents the sentence to be classified.
# returns a float that should be approximately 1. 
# If the result is less than 1, the sentence was classified as nonsense.
# If the result is >= to 1, the sentence was classified as making sense.
def sense_check_word_pair(sample):
	#classify sentence:
	#first, get tuples from input
	sample = ['SOL'] + sample.split(' ') + ['EOL']
	
	inputTuples = []
	
	for i in range(len(sample)):
		curr = sample[i]
		for j in range(1,4):
			if i < len(sample) - j:
				next = sample[i+j]
				inputTuples.append(str(str(curr) + " " + str(next)))
				
	#inputTuples is initialized
	weightedSum = 0.0
	unweightedSum = 0.0
	for i in inputTuples:
		try: 
			#if something wrong happened there, its because the dicts didn't have that feature
			#add it do the dics and then retry the calculation
			tryWeight = pairWeights[i]
		except:
			pairWeights[i] = 0
		try:
			tryWeight = pair_feature_probability[i]
		except:
			pair_feature_freq[i] = 0
			pair_feature_probability[i] = 0.0
		weightedSum = weightedSum + (pairWeights[i] * pair_feature_probability[i] + pair_feature_bias)
		unweightedSum = unweightedSum + (pair_feature_probability[i] + pair_feature_bias)
		
	result = weightedSum / unweightedSum
	#print(result)
	return result


#sense_check_pos
#takes a string as input which represents the sentence to be classified
# returns a float that should be approximately 1. 
# If the result is less than 1, the sentence was classified as nonsense.
# If the result is >= to 1, the sentence was classified as making sense.
def sense_check_pos(sample):
	#first, obtain features from input
	input = ['SOL'] + clean(sample).split(' ') + ['EOL']
	inputFeatures = []
	for j in range(len(input)):
			firstPofS = lookupPos(input[j])
			for k in range(1,4):
				if j + k < len(input):
					secondPOfS = lookupPos(input[j+k])
					feature = str(k) + ' ' + firstPofS + ' ' + secondPOfS
					if feature not in inputFeatures:
						inputFeatures.append(feature)
	
	#print(inputFeatures)
	weightedSum = 0.0
	unweightedSum = 0.0
	for i in inputFeatures:
		try:
			tryWeight = part_of_speech_weights[i]
		except:
			#if something wrong happened there, its because the dicts didn't have that feature
			#add it do the dics and then retry the calculation
			part_of_speech_weights[i] = 1.0
			P_feature[i] = 0.0
			featureFreq[i] = 0
		weightedSum = weightedSum + (part_of_speech_weights[i] * (P_feature[i] + bias))
		unweightedSum = unweightedSum + (P_feature[i] + bias)
	
	result = weightedSum / unweightedSum
	return result

#method to compare two context maps
def compareStructures(base, sample):
	
	if base == sample:
		return 1.0

	hits = 0.0
	total = 0.0
	for i in base:
		if i in sample:
			hits += 1.0
		total += 1.0
	for i in sample:
		if i in base:
			hits += 1.0
		total += 1.0
	if total == 0:
		return 0.0
	return hits/total




def createAnswerFromInput(user_input):

	inputContext = contextMap(user_input)
	inputKeywords = clean(inputContext[0]).split(' ')
	inputContext = inputContext[1]

	matching_map_sentence = ""
	matching_keywords_sentence = ""
	maxCompareValue = 0
	foundBothKW = 0
	for i in range(len(questions)):
		iMap = contextMap(questions[i])
		iKeywords = clean(iMap[0]).split(' ')
		iMap = iMap[1]
		
		#find a sentence with matching keywords
		if foundBothKW == 0 and inputKeywords[0] in iKeywords or inputKeywords[1] in iKeywords:
			matching_keywords_sentence = answers[i]
			if inputKeywords[0] in iKeywords and inputKeywords[1] in iKeywords:
				foundBothKW = 1
			
		
		compareValue = compareStructures(iMap, inputContext)
		#find arg max of compare value
		compareValue = compareStructures(iMap, inputContext)
		if compareValue > maxCompareValue:
			matching_map_sentence = answers[i]
			t = questions[i]
			maxCompareValue = compareValue
	
	
	#print("found matching keywords and question")
	#get matching map and keywords from respective sentences
	matching_map = contextMap(matching_map_sentence)[1]
	matching_keywords = clean(contextMap(matching_keywords_sentence)[0]).split(' ')
	
	#print(matching_map_sentence)
	#print(matching_keywords_sentence)
	if matching_keywords[0] == '' and matching_keywords[1] == '':
		if matching_map_sentence != '':
			return matching_map_sentence
		else:
			return "idk, hydar"
	
	#generate sentences (sample1 and sample2) from keywords
	sample1 = generate(matching_keywords[0], matching_map)
	sample1Map = contextMap(sample1)[1]
	#check sense of sample1
	sample1Sense = 0
	if sense_check_pos(sample1) >= 1.0 and sense_check_word_pair(sample1) >= 1.0:
			sample1Sense = 1

	#generate sample 2
	sample2 = generate(matching_keywords[1], matching_map)
	sample2Map = contextMap(sample2)[1]
	#check sense of sample 2
	sample2Sense = 0
	if sense_check_pos(sample2) >= 1.0 and sense_check_word_pair(sample2) >= 1.0:
			sample2Sense = 1

	#loop
	#continuously regenerate sentences if neither made sense
	#loop will only exit if one of the samples matches the conditions of 
		#1. making sense
		#2. compareStructures rates it as similar enough to the desired structure
	
	#implement a timeout
	#if the timeout is reached, use argmax of sensible sentences
	#that would be the most matching sensible sentence
	
	mostMatching = ""
	maxCompareValue = 0
	timer = time.monotonic()
	startTime = time.monotonic()
	while timer < (startTime + 20):
		compareValue1 = compareStructures(matching_map, sample1Map)
		compareValue2 = compareStructures(matching_map, sample2Map)
		if compareValue1 > maxCompareValue and sample1Sense == 1:
			mostMatching = sample1
			maxCompareValue = compareValue1
		if compareValue2 > maxCompareValue and sample2Sense == 1:
			mostMatching = sample2
			maxCompareValue = compareValue2
		
		if maxCompareValue > 0.87:
			break
		else:
			#regenerate sample1:
			sample1 = generate(matching_keywords[0], matching_map)
			sample1Map = contextMap(sample1)[1]
			#sense check sample1
			sample1Sense = 0
			if sense_check_pos(sample1) >= 1.0 and sense_check_word_pair(sample1) >= 1.0:
					sample1Sense = 1
					
			#regenerate sample2:
			sample2 = generate(matching_keywords[1], matching_map)
			sample2Map = contextMap(sample2)[1]
			#sense check sample2
			sample2Sense = 0
			if sense_check_pos(sample2) >= 1.0 and sense_check_word_pair(sample2) >= 1.0:
					sample2Sense = 1
			
			timer = time.monotonic()
	
	if mostMatching != '':
		return mostMatching
	else:
		if matching_map_sentence != '':
			return matching_map_sentence
		else:
			return "idk, hydar"
#methods defined: context map, sense check (two methods), generate random sentence, and generate sensible sentence
#now ask for user input
user_input = ""
while user_input != "-1":

	#print("Ask claire something")
	user_input = input()
	
	#result = subprocess.run(["python", "./bots/raye.py", user_input], capture_output = True)
	
	#if result == "-1":
	result = createAnswerFromInput(user_input)
	#try:
	#	print(result[0:result.index('.')])
	#except:
	print(result)
#plans
#retrain models with new data
#add rayes original data back
#add functionality for claire to do what raye does, but with lower tolerance (give known answers when prompted)




	
