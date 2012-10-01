require 'torch'
require 'xlua'

entry = entry or {}

function entry.loadSet(luafile, nEntries, inputSize, outputSize, inputOp, outputOp)
   local count = 0
   local dataset = {}
   local dim =  inputSize + outputSize
   local tensor = torch.Tensor(nEntries, dim)
   local inputOp = inputOp or function(x) return x end
   local outputOp = outputOp or function(x) return x end

   function Entry(e)
      count = count + 1
      xlua.progress(count, nEntries)
      for i=1,#e do
         tensor[count][i] = e[i]
      end
   end
   dofile(luafile)

   function dataset:normalizeInput(mean_, std_)
      local data = tensor:narrow(2, outputSize+1, dim-outputSize)
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end

   function dataset:size() return nEntries end

   local indexop = function(self, index)
      local input = tensor[index]:narrow(1, outputSize+1, dim-outputSize)
      local output = tensor[index]:narrow(1, 1, outputSize)
      local example = {inputOp(input), outputOp(output)}
      return example
   end

   setmetatable(dataset, {__index = indexop})

   return dataset
end

function entry.inputConv(iheight, iwidth)
   local inputpatch = torch.zeros(1, iheight, iwidth)
   local trans = function(input)
      local w = math.sqrt(input:nElement())
      local uinput = input:unfold(1,input:nElement(),input:nElement())
      local cinput = uinput:unfold(2,w,w)
      local h = cinput:size(2)
      local w = cinput:size(3)
      local x = math.floor((iwidth-w)/2)+1
      local y = math.floor((iheight-h)/2)+1
      inputpatch:narrow(3,x,w):narrow(2,y,h):copy(cinput)
      return inputpatch
   end
   return trans
end

function entry.outputClass(nClasses, toClassIndex)
   local labelvector = torch.zeros(nClasses)
   local trans = function(output)
      local class = toClassIndex(output)
      local label = labelvector:zero()
      label[class] = 1
      return label
   end
   return trans
end

