require 'torch'
require 'perceptron'

linreg = {}

function linreg.massage(s)
   local N = table.getn(s)
   local Nx = s[1][1]:size()[1]
   local X = torch.DoubleTensor(N, Nx+1)
   local y = torch.DoubleTensor(N)
   for r=1,N do
      y[r] = s[r][2]
      X[r][1] = 1 -- bias
      for c=1,Nx do
         X[r][c+1] = s[r][1][c]
      end
   end
   return X,y
end

function linreg.w(X, y)
   local Xt = X:transpose(1, 2)
   local Xi = torch.inverse(Xt*X)*Xt
   return Xi*y
end

function linreg.train(N, V, plot)
   opt.nn = false
   local l_true = rand_line()
   local s = rand_samples(N, l_true)

   local X, y = linreg.massage(s)
   local l_learn = linreg.w(X, y)

   if plot then viz(s, l_true, l_learn) end

   local is_missclassified = function(si)
      local pred = line_sep(l_learn, si[1])
      return pred ~= si[2]
   end
   local all_missclassified = function(s)
      local missed_s, _ = partition_samples(s, is_missclassified)
      return missed_s, table.getn(missed_s)
   end

   local _, num_missed = all_missclassified(s)
   local Ein = num_missed / N

   local es = rand_samples(V, l_true)
   local _, enum_missed = all_missclassified(es)
   local Eout = enum_missed / V

   local num_iter = 0
   local missed_s, num_missed = all_missclassified(s)
   while (num_missed > 0) do
      local ri = torch.random(num_missed)
      local si = missed_s[ri]

      l_learn = l_learn + add_bias(si[1])*si[2]

      num_iter = num_iter + 1
      if plot then viz(s, l_true, l_learn) end
      missed_s, num_missed = all_missclassified(s)
   end

   if (opt.verbose) then
      print('#' .. tostring(num_iter) .. ':' .. tostring(Ein) .. ' vs ' .. tostring(Eout))
   end

   return Ein, Eout, num_iter
end

function linreg.experiment(T, N, V)
   local avg_Ein = 0
   local avg_Eout = 0
   local avg_iter = 0
   for ti=1,T do
      if (opt.verbose) then
         print('-- trial ' .. tostring(ti) .. ' --')
      end
      local Ein,Eout,num_iter = linreg.train(N, V, opt.plot)
      avg_Ein = avg_Ein + Ein/T
      avg_Eout = avg_Eout + Eout/T
      avg_iter = avg_iter + num_iter/T
   end

   print('result for ' .. tostring(T) .. ' trials and ' .. tostring(N) .. ' samples')
   print('average Ein: ' .. tostring(avg_Ein))
   print('average Eout: ' .. tostring(avg_Eout))
   print('average numer of iterations: ' .. tostring(avg_iter))
   return avg_Ein, avg_Eout, avg_iter
end

function linreg.main()
   local avg_Ein, avg_Eout, _ = linreg.experiment(1000, 100, 1000)
   local _, _, avg_iter = linreg.experiment(1000, 10, 1000)

   print('average Ein to report: ' .. tostring(avg_Ein))
   print('average Eout to report: ' .. tostring(avg_Eout))
   print('average numer of iterations to report: ' .. tostring(avg_iter))
end