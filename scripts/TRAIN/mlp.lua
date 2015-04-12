april_print_script_header(arg)

local common = require "scripts.common"
local bagging    = common.bagging
local create_ds  = common.create_ds
local predict    = common.predict
local train_mlp  = common.train_mlp
local write_submission = common.write_submission
local mop  = matrix.op
local rnd  = random(12394)
local wrnd = random(24825)
local srnd = random(52958)
local prnd = random(24925)
local NUM_CLASSES  = 9
local ID           = assert(tonumber(arg[1]))
local HSIZE        = tonumber(arg[2] or 900)
local DEEP_SIZE    = tonumber(arg[3] or 2)
local bunch_size   = tonumber(arg[4] or 512)
local NUM_BAGS     = tonumber(arg[5] or 1)
local MAX_FEATS    = tonumber(arg[6])
local wd           = tonumber(arg[7] or 0.0)
local var          = tonumber(arg[8] or 0.2)
local mp           = tonumber(arg[9] or 999)
local feats_name   = arg[10] or "std"
local opt          = arg[11] or "adadelta"

print("# hsize deep_size bunch_size num_bags max_feats wd var mp feats")
print("#", HSIZE, DEEP_SIZE, bunch_size, NUM_BAGS,
      MAX_FEATS, wd, var, mp, feats_name)

local max_epochs = 1000

local optimizer = opt
local options = {
  --learning_rate = 0.4,
  --momentum = 0.9,
  max_norm_penalty = (mp < 999 and mp) or nil
}
for i=12,#arg do
  local k,v = arg[i]:match("([^%=]+)%=([^%=]+)")
  options[k] = tonumber(v)
end

print("# max_epochs", max_epochs)
print("# optimizer", optimizer)
print("# options", iterator(pairs(options)):concat("="," "))

local bagging_iteration=0
local function train(train_data, train_labels, val_data, val_labels)
  local HSIZE = HSIZE
  bagging_iteration = bagging_iteration + 1
  if bagging_iteration > 1 then
    HSIZE = math.round( HSIZE * (prnd:rand(1.5) + 0.25) )
  end
  print("# HSIZE", HSIZE)
  local isize = train_data:dim(2)
  local model = ann.mlp.all_all.generate("%d inputs %d relu 0.5 dropout(#1) %d relu 0.5 dropout(#1) %d log_softmax"%
                                           { isize, HSIZE, HSIZE, NUM_CLASSES },
                                         { prnd })
  local trainer = trainable.supervised_trainer(model,
                                               ann.loss.multi_class_cross_entropy(),
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
  
  local train_in_ds,train_out_ds = create_ds(train_data, train_labels,
                                             NUM_CLASSES)
  local val_in_ds,val_out_ds = create_ds(val_data, val_labels, NUM_CLASSES)
  
  local train_in_ds = dataset.perturbation{ dataset  = train_in_ds,
                                            random   = prnd,
                                            variance = var }
  
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

local train_data = matrix.fromTabFilename("DATA/train_feats.%s.split.mat.gz"%{feats_name})
local train_labels = matrix.fromTabFilename("DATA/train_labels.split.mat.gz")
local val_data = matrix.fromTabFilename("DATA/val_feats.%s.split.mat.gz"%{feats_name})
local val_labels = matrix.fromTabFilename("DATA/val_labels.split.mat.gz")

print("# DATA SIZES", train_data:dim(1), train_data:dim(2),
      val_data:dim(1), val_data:dim(2))

local bagging_models = bagging(NUM_CLASSES, NUM_BAGS, MAX_FEATS, rnd,
                               train_data, train_labels,
                               val_data, val_labels,
                               train, predict_mlp,
                               "ID_%03d.validation.mlp.csv"%{ID})

-----------------------------------------------------------------------------

local test_data = matrix.fromTabFilename("DATA/test_feats.%s.split.mat.gz"%{feats_name})
local test_p = predict_mlp(bagging_models, test_data)
print(test_p)

write_submission("ID_%03d.test.mlp.csv"%{ID}, test_p)
