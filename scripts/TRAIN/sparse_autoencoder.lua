april_print_script_header(arg)

local common = require "scripts.common"
local bagging    = common.bagging
local create_ds  = common.create_ds
local predict    = common.predict
local train_mlp  = common.train_mlp
local actf       = ann.components.actf
local components = ann.components
local mop  = matrix.op
local rnd  = random(582576)
local wrnd = random(2345)
local prnd = random(83587)
local srnd = random(102384)
local result_name  = arg[1] or "sae"
local HSIZE        = tonumber(arg[2] or 900)
local DEEP         = tonumber(arg[3] or 1)
local bunch_size   = tonumber(arg[4] or 512)
local wd           = tonumber(arg[5] or 0.0)
local var          = tonumber(arg[6] or 0.0)
local mp           = tonumber(arg[7] or 999)
local feats_name   = arg[8] or "int600"
local opt          = arg[9] or "adadelta"

local max_epochs = 10000

local optimizer = opt
local options = {
  --learning_rate = 0.4,
  --momentum = 0.9,
  max_norm_penalty = (mp < 999 and mp) or nil
}
for i=10,#arg do
  local k,v = arg[i]:match("([^%=]+)%=([^%=]+)")
  assert(k and v, "Unable to parse given option key=value")
  options[k] = tonumber(v)
end

print("# hsize deep bunch_size wd var mp feats")
print("#", HSIZE, DEEP_SIZE, bunch_size, wd, var, mp, feats_name)
print("# max_epochs", max_epochs)
print("# optimizer", optimizer)
print("# options", iterator(pairs(options)):concat("="," "))

for i=1,DEEP do
  collectgarbage("collect")
  local train_data = matrix.fromTabFilename("DATA/train_feats.%s.split.mat.gz"%{feats_name})
  local val_data = matrix.fromTabFilename("DATA/val_feats.%s.split.mat.gz"%{feats_name})
  
  print("# HSIZE", HSIZE)
  local isize = train_data:dim(2)
  local model = ann.components.stack():
    push(
      components.hyperplane{ input=isize, output=HSIZE, dot_product_weights="w1", bias_weights="b1" },
      actf.sparse_logistic{ sparsity=0.01, penalty=0.1 },
      components.hyperplane{ input=HSIZE, output=isize, dot_product_weights="w1", bias_weights="b2", transpose = true },
      actf.linear()
    )
  local trainer = trainable.supervised_trainer(model,
                                               ann.loss.mse(),
                                               bunch_size,
                                               ann.optimizer[optimizer]())
  trainer:build()
  trainer:randomize_weights{
    random = wrnd,
    inf = -3,
    sup = 3,
    use_fanin = true,
    use_fanout = true,
  }
  for _,b in trainer:iterate_weights("b.*") do b:zeros() end
  trainer:set_layerwise_option("w.*", "weight_decay", wd)
  for name,value in pairs(options) do
    trainer:set_option(name, value)
  end

  local train_in_ds = dataset.matrix(train_data)
  local val_in_ds = dataset.matrix(val_data)

  if var > 0.0 then
    train_in_ds = dataset.perturbation{ dataset  = train_in_ds,
                                        random   = prnd,
                                        variance = var }
  end

  local best = train_mlp(trainer,
                         max_epochs,
                         { input_dataset = train_in_ds,
                           output_dataset = train_in_ds,
                           shuffle = srnd,
                           replacement = math.max(2560, bunch_size) },
                         { input_dataset = val_in_ds,
                           output_dataset = val_in_ds })

  best:get_component():pop():pop()
  best:build()

  local function write_code(data, filename)
    print("# Writing", filename)
    local f = io.open(filename, "w")
    for pat in trainable.dataset_multiple_iterator{
      datasets = { dataset.matrix(data) },
      bunch_size = 512,
    } do
      local out = best:calculate(pat)
      out:write(f, { tab=true })
    end
    f:close()
  end

  write_code(train_data, "DATA/train_feats.%s%d.split.mat.gz"%{ result_name, HSIZE })
  write_code(val_data, "DATA/val_feats.%s%d.split.mat.gz"%{ result_name, HSIZE })

  train_data = nil
  val_data = nil
  collectgarbage("collect")

  local test_data = matrix.fromTabFilename("DATA/test_feats.%s.split.mat.gz"%{feats_name})
  write_code(test_data, "DATA/test_feats.%s%d.split.mat.gz"%{ result_name, HSIZE })
  
  feats_name = "%s%d"%{ result_name, HSIZE }
end
