import math
from collections import defaultdict, Counter


class trelNode:
    def __init__(self, p, ptr, tag, w):
        self.p = p
        self.bkptr = ptr
        self.tag = tag
        self.word = w


def countTags(sentences):
    tagWordCounts = defaultdict(Counter)  # Use defaultdict to eliminate the need for checking if a word exists
    countOfTags = Counter()
    IniTagCount = Counter()
    countTagTrans = defaultdict(Counter)  # Use defaultdict to eliminate the need for checking if a tag exists
    hapxCounts = Counter()
    hapxTC = 0

    for sentence in sentences:
        # Use itertools to efficiently iterate over pairs of words and tags
        for (word, tag), (_, nextTag) in zip(sentence, sentence[1:]):
            tagWordCounts[word].update([tag])  # Update tagWordCounts for the current word

            # Update countTagTrans for transitions from the current tag to the next tag
            countTagTrans[tag].update([nextTag])

            countOfTags.update([tag])  # Update countOfTags for the current tag

            if sentence.index((word, tag)) == 0:
                IniTagCount.update([tag])  # Update IniTagCount for the first tag in the sentence

    # Use a set to keep track of unique word-tag pairs for each tag
    uniqWordTagPairs = defaultdict(set)
    for sentence in sentences:
        for (word, tag) in sentence:
            uniqWordTagPairs[tag].add(word)

    # Add the 'UNKWORD' key to tagWordCounts with an empty Counter
    tagWordCounts['UNKWORD'] = Counter()

    # storeWordTagHapx = []
    for tag in countOfTags:
        for word in uniqWordTagPairs[tag]:
            if tagWordCounts[word][tag] == 1:
                hapxCounts.update([tag])
                # storeWordTagHapx.append([word,tag])
                hapxTC += 1

    #     # Specify the output file name for storing Word-Tag pairs of hapax words
    # hapax_output_file = "hapax_word_tag_pairs.txt"

    # # Open the file in write mode and write the Word-Tag pairs
    # with open(hapax_output_file, "w") as file:
    #     for cum in storeWordTagHapx:
    #         file.write(f'{cum[0]},{cum[1]}\n')

    # print(f"Hapax Word-Tag pairs written to {hapax_output_file}")

    return tagWordCounts, countOfTags, IniTagCount, countTagTrans, hapxCounts, hapxTC


def calculateProbabilities(train, tagWordCounts, countOfTags, IniTagCount, countTagTrans, hapxCounts, hapxTC):
    emissionProb = dict()
    transitionProb = dict()
    initialProb = dict()
    hapxProb = dict()

    totalUniqWords = len(tagWordCounts)     # Number of unique words in the training set
    totalTags = len(countOfTags) + 1           # Number of tags in the training set, + 1 to account for 'UNKNOWNTAG'


    alpha = 1  # Smoothing Constant
    # hapxProb builder
    for tag in countOfTags:
        hapxProb[tag] = (hapxCounts[tag] + alpha) / (hapxTC + alpha * totalTags)

    # emissionProb builder
    for word in tagWordCounts:
        emissionProb[word] = dict()
        for tag in countTagTrans:
            emissionProb[word][tag] = (tagWordCounts[word][tag] + (alpha * hapxProb[tag])) / (countOfTags[tag] + ( (alpha * hapxProb[tag]) * totalUniqWords))  # Numerator will be alpha for 'UNKNOWN'

    # transitionProb builder
    for prevTag in countTagTrans:
        transitionProb[prevTag] = dict()
        for nxtTag in countTagTrans:
            transitionProb[prevTag][nxtTag] = (countTagTrans[prevTag][nxtTag] + alpha) / (countOfTags[prevTag] + alpha * totalTags)

    # initialProb builder
    for tag in countTagTrans:
        initialProb[tag] = (IniTagCount[tag] + alpha) / (len(train) + alpha * totalTags)

    return emissionProb, transitionProb, initialProb

def trellisAndBacktrack(test, tagWordCounts, emissionProb, transitionProb, initialProb):
    predicts = []

    # Iterate over sentences in the test set
    for sentence in test:
        # Initialize previous word-tag nodes
        previousWordTagNode = []

        # Iterate over words in the sentence
        for idx, word in enumerate(sentence):
            # Initialize current word-tag nodes
            NodesCurrentWordTag = []

            if word in tagWordCounts:
                if idx == 0:
                    for tag in tagWordCounts[word]:
                        p = math.log(initialProb[tag]) + math.log(emissionProb[word][tag])
                        NodesCurrentWordTag.append(trelNode(p, None, tag, word))
                else:
                    for tag in tagWordCounts[word]:
                        p = math.log(emissionProb[word][tag]) + max([node.p + math.log(transitionProb[node.tag][tag]) for node in previousWordTagNode])
                        maxNode = max(previousWordTagNode, key=lambda node: node.p + math.log(transitionProb[node.tag][tag]))  # for back pointer
                        NodesCurrentWordTag.append(trelNode(p, maxNode, tag, word))
            else:
                if idx == 0:
                    for tag in transitionProb:
                        p = math.log(initialProb[tag]) + math.log(emissionProb['UNKWORD'][tag])
                        NodesCurrentWordTag.append(trelNode(p, None, tag, word))
                else:
                    for tag in transitionProb:
                        p = math.log(emissionProb['UNKWORD'][tag]) + max([node.p + math.log(transitionProb[node.tag][tag]) for node in previousWordTagNode])
                        maxNode = max(previousWordTagNode, key=lambda node: node.p + math.log(transitionProb[node.tag][tag]))  # for back pointer
                        NodesCurrentWordTag.append(trelNode(p, maxNode, tag, word))

            previousWordTagNode = NodesCurrentWordTag

        # Backtracking for each sentence
        tempRevSen = []
        CurrentNode = max(previousWordTagNode, key=lambda node_: node_.p)

        while CurrentNode is not None:
            tempRevSen.append((CurrentNode.word, CurrentNode.tag))
            CurrentNode = CurrentNode.bkptr

        tempRevSen.reverse()

        predicts.append(tempRevSen)

    return predicts

def viterbi_2(train, test):
    tagWordCounts, countOfTags, IniTagCount, countTagTrans, hapxCounts, hapxTC = countTags(train)
    
    emissionProb, transitionProb, initialProb = calculateProbabilities(train, tagWordCounts, countOfTags, IniTagCount, countTagTrans, hapxCounts, hapxTC)

    predicts = trellisAndBacktrack(test, tagWordCounts, emissionProb, transitionProb, initialProb)

    return predicts
