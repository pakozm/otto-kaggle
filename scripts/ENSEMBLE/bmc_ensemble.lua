local common = require "scripts.common"
local load_ensemble_model_from_csv = common.load_ensemble_model_from_csv
local write_submission = common.write_submission
--
local ID = table.remove(arg,1)
local N = #arg

assert(not ID:find("/"))

local function get_filenames(arg)
  local validation,test = {},{}
  for i=1,#arg do
    local va,te = glob(arg[i].."*validation*"),glob(arg[i].."*test*")
    assert(#va == 1 and #te == 1)
    validation[#validation+1],test[#test+1] = va[1],te[1]
  end
  return validation,test
end

local validation,test = get_filenames(arg)
local val_tgt = matrix.fromTabFilename("DATA/val_labels.split.mat.gz")
local val_ensemble = load_ensemble_model_from_csv(validation, val_tgt)
local test_ensemble = load_ensemble_model_from_csv(test)

local rnd = random(12384)
local NaN = mathcore.limits.float.quiet_NaN()
local SAMPLES = 2000 --500
local weights = matrix(N):fill(1/N)
---------------- BMC -----------------------------------------------------
local w1 = weights
local wr = w1:clone() -- result weights
local z = -math.huge
local sum = 0
weights:zeros()
for i=1,SAMPLES do
  -- sample from uniform dirichlet
  for j=1,N do w1[j] = -math.log(rnd:rand()) end
  w1:scal(1/w1:sum())
  -- loss
  local loss = val_ensemble:compute_loss(w1)
  -- log-likelihood
  local loglh = -val_tgt:dim(1) * loss
  if loglh > z then -- for numerical stability
    wr:scal( math.exp(z - loglh) )
    z = loglh
  end
  local w = math.exp(loglh - z)
  wr[{}] = wr * sum / (sum + w) + w * w1
  sum = sum + w
end
wr:scal(1/wr:sum())
w1:copy(wr)
--
print(table.concat(weights:toTable()," "))
print("# VA", val_ensemble:compute_loss(weights))
--
local test_p = test_ensemble:calculate(weights)
write_submission("ID_%03d.test.bmc.csv"%{ID}, test_p)
