require 'torch'
require 'perceptron'

linreg = {}

function rand_samples_fun(N, f)
   local s = {}
   for i=1,N do
      local x = rand_point()
      local y = f(x)
      s[i] = {x,y}
   end
   return s
end

function samples_rand_noise(s, r)
   for _,si in ipairs(s) do
      if (torch.rand(1)[1] < r) then
         si[2] = -si[2]
      end
   end
   return s
end

function samples_nonlinear_trans(s)
   for _,si in ipairs(s) do
      local x = si[1]
      si[1] = torch.DoubleTensor({x[1],x[2],x[1]*x[2],x[1]*x[1],x[2]*x[2]})
   end
   return s
end

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

function linreg.experiment(T, N, V, train_fun)
   train_fun = train_fun or linreg.train
   local avg_Ein = 0
   local avg_Eout = 0
   local avg_iter = 0
   for ti=1,T do
      if (opt.verbose) then
         print('-- trial ' .. tostring(ti) .. ' --')
      end
      local Ein,Eout,num_iter = train_fun(N, V, opt.plot)
      avg_Ein = avg_Ein + Ein/T
      avg_Eout = avg_Eout + Eout/T
      avg_iter = num_iter/T + avg_iter
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

function linreg.train2(N, V, plot)
   opt.nn = false
   local noise_r = 0.1
   local f = function(x)
      y = x[1]*x[1] + x[2]*x[2] - 0.6
      if (y < 0) then return -1 else return 1 end
   end
   local l_dummy = perceptron() -- not used
   local s = samples_rand_noise(rand_samples_fun(N, f), noise_r)

   local X, y = linreg.massage(s)
   local l_learn = linreg.w(X, y)

   if plot then viz(s, l_dummy, l_learn) end

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

   samples_nonlinear_trans(s)
   X, y = linreg.massage(s)
   l_learn = linreg.w(X, y)

   local es = samples_rand_noise(rand_samples_fun(V, f), noise_r)
   samples_nonlinear_trans(es)
   local _, enum_missed = all_missclassified(es)
   local Eout = enum_missed / V

   if (opt.verbose) then
      print(tostring(Ein) .. ' vs ' .. tostring(Eout))
      print(l_learn)
   end
   return Ein, Eout, l_learn
end

function linreg.main2()
   local avg_Ein, avg_Eout, avg_w = linreg.experiment(1000, 1000, 1000, linreg.train2)

   print('average Ein to report: ' .. tostring(avg_Ein))
   print('average Eout to report: ' .. tostring(avg_Eout))
   print(avg_w)
   return avg_w
end