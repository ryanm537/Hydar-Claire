#train weights for use in sense-nonsense test

#extract dictionary from text file and put it in an array
#extract dictionary from text file and put it in an array
f = str(open("parts_of_speech.txt", "r").read()).split('\n')
partsOfSpeech = {}
for i in f:
	i = i.split('#')
	if len(i[1]) > 1:
		i[1] = i[1][0:1] + '+'
	partsOfSpeech[i[0]] = i[1]

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



#initialize frequencies of each unique part of speech
posFreq = {}
posFreq[lookupPos('SOL')] = 0
posFreq[lookupPos('EOL')] = 0
#file = str(open('raye_responses.txt', 'r', encoding = 'utf-8').read()).split('\n')
file = str(open('training_data_4.txt', 'r', encoding = 'utf-8').read()).split('\n')
#for i in file2:
#	file.append(i)

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


bias = (P_feature[min(P_feature, key=P_feature.get)]) / float(len(posFreq))

#add zero probabilities (and frequencies) for features that do not exist in the data but technically could
for k1, v1 in posFreq.items():
	for k2, v2 in posFreq.items():	
		for k in range(1, 4):
			possibleFeature = str(k) + ' ' + k1 + ' ' + k2
			if possibleFeature not in featureFreq:
				featureFreq[possibleFeature] = 0
				P_feature[possibleFeature] = 0.0

#initialize weights dict
#all weights start out at 1
weights = {}
for k, v in featureFreq.items():
	weights[k] = 1.0


#algorithm
#rate a sentence as either making sense or not making sense
#create this rating by taking the sum(weighted probabilities) / sum(unweighted probabilities)
#if that result is < 1, the sentence will be rated as nonsense, and if it is >= 1, it will be rated as sense.
#if it incorrectly marked a sensible sentence as nonsense, increase weights for all pairs  in that sentence
#if it incorrectly maked a nonsense sentence as sense, decrease weights for all pairs in that sentence

training_data = str(open('training_data5.txt', 'r', encoding = 'utf-8').read()).split('\n')

num_iter = 200
for iterations in range(num_iter):

	correct = 0
	incorrect = 0

	for line in training_data:

		sense = int(line[0])
		input = line[1:len(line)]
		#print(run(input)[1])
		#first, obtain features from input
		input = ['SOL'] + clean(input).split(' ') + ['EOL']
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
				tryWeight = weights[i]
			except:
				#if something wrong happened there, its because the dicts didn't have that feature
				#add it do the dics and then retry the calculation
				weights[i] = 1.0
				P_feature[i] = 0.0
				featureFreq[i] = 0
			weightedSum = weightedSum + (weights[i] * (P_feature[i] + bias))
			unweightedSum = unweightedSum + (P_feature[i] + bias)
		
		result = weightedSum / unweightedSum
		if result < 1:
			#sentence was rated as nonsense
			if sense == 0:
				#sentence is indeed nonsense
				correct += 1
			else:
				#sentence actually made sense
				incorrect += 1
				#raise all weights
				for i in inputFeatures:
					weights[i] = weights[i] * 1.001
		else:
			#sentence was rated as sense
			if sense == 1:
				#sentence is indeed sense
				correct += 1
			else:
				#sentence actually did not make sense
				incorrect += 1
				#lower all weights
				for i in inputFeatures:
					weights[i] = weights[i] * 0.999

	accuracy = float(correct) / float(correct + incorrect)
	print("accuracy: " + str(accuracy))

#write results to a file
newFile = open("claire_part_of_speech_weights.txt", 'w')
for k, v in weights.items():
	newFile.write(str(k) + ' ' + str(v) + '\n')




