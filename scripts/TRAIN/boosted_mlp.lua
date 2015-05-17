mathcore.set_use_cuda_default( util.is_cuda_available() )
april_print_script_header(arg)

local common = require "scripts.common"
local adaboost   = common.adaboost
local bootstrap  = common.bootstrap
local create_ds  = common.create_ds
local gradient_boosting = common.gradient_boosting
local predict    = common.predict
local train_mlp  = common.train_mlp
local write_submission = common.write_submission
local mop  = matrix.op
local rnd  = random(12394)
local wrnd = random(24825)
local srnd = random(52958)
local prnd = random(24925)
local NUM_CLASSES  = 9
local ID = assert(tonumber(arg[1]))
local HSIZE        = tonumber(arg[2] or 900)
local DEEP_SIZE    = tonumber(arg[3] or 2)
local bunch_size   = tonumber(arg[4] or 512)
local NUM_BAGS     = tonumber(arg[5] or 50)
local ACTF         = "prelu"
local input_drop   = 0.2
local use_dropout  = true

local ver="std"

local max_epochs = 10000

local optimizer = "adadelta"
local options = {
  -- learning_rate = 0.0000,
  -- momentum = 0.9,
}

local function train(train_data, train_labels, val_data, val_labels)
  print("# HSIZE", HSIZE)
  local isize = train_data:dim(2)
    local topology = { "%d inputs"%{isize} }
  if use_dropout then
    table.insert(topology, "dropout{prob=#2,random=#1}")
  end
  for i=1,DEEP_SIZE do
    table.insert(topology, "%d %s"%{HSIZE,ACTF})
    if use_dropout then
      table.insert(topology, "dropout{prob=0.5,random=#1}")
    end
  end
  table.insert(topology, "%d log_softmax"%{NUM_CLASSES})
  local topology = table.concat(topology, " ")
  print("# MLP", topology)
  local model = ann.mlp.all_all.generate(topology, { prnd, input_drop })
  local trainer = trainable.supervised_trainer(model,
                                               ann.loss.multi_class_cross_entropy(),
                                               bunch_size,
                                               ann.optimizer[optimizer]())
  trainer:build()
  trainer:randomize_weights{
    name_match = "[bw].*",
    random = wrnd,
    inf = -3,
    sup = 3,
    use_fanin = true,
    use_fanout = true,
  }
  for _,b in trainer:iterate_weights("b.*") do b:zeros() end
  for _,a in trainer:iterate_weights("a.*") do a:fill(0.25) end
  trainer:set_layerwise_option("w.*", "weight_decay", 0.00)
  for name,value in ipairs(options) do
    trainer:set_option(name, value)
  end
  
  local train_in_ds,train_out_ds = create_ds(train_data, train_labels,
                                             NUM_CLASSES)
  local val_in_ds,val_out_ds = create_ds(val_data, val_labels, NUM_CLASSES)
  
  local train_in_ds = dataset.perturbation{ dataset  = train_in_ds,
                                            random   = prnd,
                                            variance = 0.0 }
  
  local best = train_mlp(trainer,
                         max_epochs,
                         { input_dataset = train_in_ds,
                           output_dataset = train_out_ds,
                           shuffle = srnd,
                           replacement = math.max(2560, bunch_size) },
                         { input_dataset = val_in_ds,
                           output_dataset = val_out_ds })
  return best
end

local predict_mlp = function(models, data)
  local func = function(model, data)
    return model:calculate(data):exp()
  end
  local p = predict(models, data, func)
  return p
end

local train_data = matrix.fromTabFilename("DATA/train_feats.%s.split.mat.gz"%{ver})
local train_labels = matrix.fromTabFilename("DATA/train_labels.split.mat.gz"%{ver})
local val_data = matrix.fromTabFilename("DATA/val_feats.%s.split.mat.gz"%{ver})
local val_labels = matrix.fromTabFilename("DATA/val_labels.split.mat.gz"%{ver})

print("# DATA SIZES", train_data:dim(1), train_data:dim(2),
      val_data:dim(1), val_data:dim(2))

--local models = adaboost(method,
--                        NUM_CLASSES, NUM_BAGS, rnd,
--                        train_data, train_labels,
--                        val_data, val_labels,
--                        train, predict_mlp)

local models = gradient_boosting(ann.loss.multi_class_cross_entropy(),
                                 1.0, 3.0,
                                 NUM_CLASSES, NUM_BAGS, rnd,
                                 train_data, train_labels,
                                 val_data, val_labels,
                                 train, predict_mlp,
                                 "ID_%03d.validation.bmlp.csv"%{ID})

-----------------------------------------------------------------------------

local test_data = matrix.fromTabFilename("DATA/test_feats.%s.split.mat.gz"%{ver})
local test_p = predict_mlp(models, test_data)
print(test_p)

write_submission("ID_%03d.test.bmlp.csv"%{ID}, test_p)
