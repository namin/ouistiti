require 'torch'

function main()
   local ntrials = 100000
   local ncoins = 1000
   local ntosses = 10
   local headp = 0.5

   local v1=0
   local vrand=0
   local vmin=0
   for trial=1,ntrials do
      if (trial%1000==0) then
         print(trial .. '...')
         local r = ntrials/trial
         print('avg v1 ' .. v1*r)
         print('avg vrand ' .. vrand*r)
         print('avg vmin ' .. vmin*r)
      end
      local c1 = 1
      local crand = torch.random(ncoins)
      local cmin = 1
      local h1 = 1
      local hrand = 1
      local hmin = 1
      for coin=1,ncoins do
         local heads = torch.rand(ntosses):lt(headp):sum()
         if (coin == c1) then h1 = heads end
         if (coin == crand) then hrand = heads end
         if (heads < hmin) then cmin = coin; hmin = heads end
      end
      v1 = v1 + (h1/(ntrials*ntosses))
      vrand = vrand + (hrand/(ntrials*ntosses))
      vmin = vmin + (hmin/(ntrials*ntosses))
   end

   print('final avg v1 ' .. v1)
   print('final avg vrand ' .. vrand)
   print('final avg vmin ' .. vmin)
end
if arg then main() end