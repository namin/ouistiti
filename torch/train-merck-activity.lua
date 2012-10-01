require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'entry'
require 'kaggle-merck-activity'

----------------------------------------------------------------------
-- parse command-line options
--
dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('Merck Activity Training')
cmd:text()
cmd:text('Options:')
cmd:option('-save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-full', false, 'use full dataset (60,000 samples)')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 1, 'nb of threads')
cmd:option('-nHidden', 30, 'size of hidden layer')
cmd:option('-learningRate', 0.01, 'learning rate')
cmd:option('-learningRateDecay', 0.001, 'learning rate decay')
cmd:option('-maxIteration', 100, 'max iterations')
cmd:option('-shuffleIndices', false, 'shuffle indices')
cmd:text()
opt = cmd:parse(arg)

-- fix seed
torch.manualSeed(opt.seed)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

----------------------------------------------------------------------
-- define model to train

if opt.network == '' then
   model = nn.Sequential()
   model:add(nn.Linear(kaggle.inputSize, opt.nHidden))
   model:add(nn.Tanh())
   model:add(nn.Linear(opt.nHidden, opt.nHidden))
   model:add(nn.Tanh())
   model:add(nn.Linear(opt.nHidden,kaggle.outputSize))
else
   print('<trainer> reloading previously trained network')
   model = torch.load(opt.network)
end

parameters,gradParameters = model:getParameters()

-- verbose
print('<merck> using model:')
print(model)

-- criterion
criterion = nn.MSECriterion()

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   nbTrainingPatches = kaggle.trainSize
   nbTestingPatches = kaggle.testSize
else
   nbTrainingPatches = math.floor(0.05*kaggle.trainSize)
   nbTestingPatches = math.floor(0.05*kaggle.testSize)
   print('<warning> sampling to train quickly')
end

-- create training set and normalize
labelledData = kaggle.loadTrainSet(nbTrainingPatches)
trainData, validationData = entry.split(labelledData, 0.1)

-- create test set and normalize
testData = kaggle.loadTestSet(nbTestingPatches)

----------------------------------------------------------------------
-- define training and testing functions
--

trainer = nn.StochasticGradient(model, criterion)

function train(dataset)
   trainer.learningRate = opt.learningRate
   trainer.learningRateDecay = opt.learningRateDecay
   trainer.maxIteration = opt.maxIteration
   trainer.shuffleIndices = opt.shuffleIndices
   trainer:train(dataset)
end

function validate(dataset)
   local meanErr = 0
   local n = dataset:size()
   for i=1,n do
      local output = model:forward(dataset[i][1])
      local err = criterion:forward(output, dataset[i][2])
      meanErr = meanErr + (err / n)
   end
   return meanErr
end

function predict(input)
   local output = model:forward(input)
   return (output:squeeze()*kaggle.stdOutput)+kaggle.meanOutput
end

function output(dataset, filename)
   local f = io.open(filename, 'w')
   for t = 1,dataset:size() do
      -- disp progress
      xlua.progress(t, dataset:size())
      f:write(predict(dataset[t]) .. "\n")
   end
   f:close()
end

----------------------------------------------------------------------
-- and train!
--
do
   local i = 1
   local prevValidErr = 100
   local validErr = 100
   while prevValidErr >= validErr do
      -- train/test
      train(trainData)
      prevValidErr, validErr = validErr, validate(validationData)
      print(i .. ". validation error " .. validErr)
      output(testData, kaggle.outPrefix .. i .. '.csv')
      i = i + 1
   end
end
