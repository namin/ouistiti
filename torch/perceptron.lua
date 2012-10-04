require 'torch'

function rand_point()
   return (torch.rand(2)*2)-1
end

function rand_line()
   local p1 = rand_point()
   local p2 = rand_point()
   local a = (p2[2] - p1[2])/(p2[1] - p1[1])
   local b = p1[2] - a*p1[1]
   local l = torch.DoubleTensor({b, a, -1})
   return l
end

function add_bias(p)
   local x = torch.DoubleTensor({1, p[1], p[2]})
   return x
end

function line_sep(l, p)
   local x = add_bias(p)
   local i = l:dot(x)
   if (i > 0) then return 1 else return -1 end
end

function line_samples(l, N)
   local s = {}
   local np = 0
   while (np == 0 or np == N) do
      s = {}
      np = 0
      for i=1,N do
         local p = rand_point()
         local c = line_sep(l, p)
         if c == 1 then np = np + 1 end
         s[i] = {p, c}
      end
   end
   return s
end

function line_plot(l, w, s)
   local x = torch.linspace(-1, 1)
   local tpx = {}
   local tpy = {}
   local tnx = {}
   local tny = {}
   for i=1,table.getn(s) do
      if (s[i][2] == 1) then
         table.insert(tpx, s[i][1][1])
         table.insert(tpy, s[i][1][2])
      else
         table.insert(tnx, s[i][1][1])
         table.insert(tny, s[i][1][2])
      end
   end
   local px = torch.DoubleTensor(tpx)
   local py = torch.DoubleTensor(tpy)
   local nx = torch.DoubleTensor(tnx)
   local ny = torch.DoubleTensor(tny)
   gnuplot.plot({nx,ny},
                {px, py},
                {x,(x*l[2]+l[1])/(-l[3]), '-'},
                {x,(x*w[2]+w[1])/(-w[3]), '-'})
end

function run(N, doplot)
   local l = rand_line()
   local s = line_samples(l, N)
   local w = torch.zeros(3)
   w[3] = -1
   if doplot then line_plot(l, w, s) end

   function iter()
      local miss = {}
      for i=1,N do
         if line_sep(w, s[i][1]) ~= s[i][2] then
            table.insert(miss, s[i])
         end
      end
      return miss
   end

   local n = 0
   local miss = iter()
   local nmiss = table.getn(miss)
   while (nmiss > 0) do
      r = torch.random(nmiss)
      a = miss[r]
      w = w + add_bias(a[1])*a[2]
      n = n+1
      if doplot then line_plot(l, w, s) end
      miss = iter()
      nmiss = table.getn(miss)
   end

   local pr = torch.acos(torch.abs(l:dot(w)/(l:norm()*w:norm())))/(2*torch.acos(0))
   print(n)
   print(pr)
   return {n, pr}
end

function run_many(trials, N, doplot)
   local na = 0
   local pa = 0
   for i=1,trials do
      print('trial -- ' .. i)
      local r = run(N, doplot)
      na = na + r[1]/trials
      pa = pa + r[2]/trials
   end
   print(na)
   print(pa)
   return {na, pa}
end