require 'entry'

kaggle = kaggle or {}

kaggle.outPrefix = 'out-act7-'
kaggle.trainFile = 'act7train.lua'
kaggle.testFile = 'act7test.lua'
kaggle.trainSize = 1569
kaggle.testSize = 523
kaggle.inputSize = 4505
kaggle.outputSize = 1

function kaggle.loadTrainSet(size)
   local trainSize = size or kaggle.trainSize
   local dataset = entry.loadSet(kaggle.trainFile, trainSize, kaggle.inputSize, kaggle.outputSize)
   kaggle.meanInput, kaggle.stdInput = dataset:normalizeInputGlobal()
   kaggle.meanOutput, kaggle.stdOutput = dataset:normalizeOutputGlobal()
   return dataset
end

function kaggle.loadTestSet(size)
   local testSize = size or kaggle.testSize
   local dataset = entry.loadSet(kaggle.testFile, testSize, kaggle.inputSize, 0)
   dataset:normalizeInputGlobal(kaggle.meanInput, kaggle.stdInput)
   return dataset
end