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

local softmax = ann.components.actf.softmax()
local weights = matrix(N):fill(1/N)
local opt = ann.optimizer.simplex()
opt:set_option("epsilon", 1e-6)
opt:set_option("max_iter", 1000)
opt:set_option("rand", random(245924))
opt:set_option("tol", 1e-6)
opt:execute(
  function(w)
    return val_ensemble:compute_loss(softmax:forward(w.w))
  end, {w=weights})
--
local weights = softmax:forward(weights)
print(table.concat(weights:toTable()," "))
print("# VA", val_ensemble:compute_loss(weights))
--
local test_p = test_ensemble:calculate(weights)
write_submission("ID_%03d.test.ensemble.csv"%{ID}, test_p)
