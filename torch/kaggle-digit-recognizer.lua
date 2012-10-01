require 'entry'

kaggle = kaggle or {}

kaggle.trainFile = 'train.lua'
kaggle.testFile = 'test.lua'
kaggle.trainSize = 42000
kaggle.testSize = 28000
kaggle.imageHeight = 28
kaggle.imageWidth = 28
kaggle.inputSize = kaggle.imageHeight * kaggle.imageWidth
kaggle.outputSize = 1

kaggle.inputOp = entry.inputConv(kaggle.imageHeight, kaggle.imageWidth)
kaggle.outputOp = entry.outputClass(10, function(x) return x:squeeze()+1 end)

function kaggle.loadTrainSet(size)
   local trainSize = size or kaggle.trainSize
   local dataset = entry.loadSet(kaggle.trainFile, trainSize, kaggle.inputSize, kaggle.outputSize, kaggle.inputOp, kaggle.outputOp)
   kaggle.mean, kaggle.std = dataset.normalizeInput()
   return dataset
end

function kaggle.loadTestSet(size)
   local testSize = size or kaggle.testSize
   local dataset = entry.loadSet(kaggle.testFile, testSize, kaggle.inputSize, 0, kaggle.inputOp)
   dataset.normalizeInput(kaggle.mean, kaggle.std)
   return dataset
end
