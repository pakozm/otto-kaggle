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
local HSIZE        = tonumber(arg[1] or 256)
local DEEP_SIZE    = tonumber(arg[2] or 2)
local bunch_size   = tonumber(arg[3] or 512)
local NUM_BAGS     = tonumber(arg[4] or 1)
local MAX_FEATS    = tonumber(arg[5])
local wd           = tonumber(arg[6] or 0.00)
local var          = tonumber(arg[7] or 0.2)

local max_epochs = 10000

local optimizer = "adadelta"
local options = {
  --learning_rate = 0.4,
  --momentum = 0.4,
}

local bagging_iteration=0
local function train(train_data, train_labels, val_data, val_labels)
  local HSIZE = HSIZE
  bagging_iteration = bagging_iteration + 1
  if bagging_iteration > 1 then
    HSIZE = math.round( HSIZE * (prnd:rand(1.5) + 0.25) )
  end
  print("# HSIZE", HSIZE)
  local isize = train_data:dim(2)
  local model = ann.components.stack()
  for i=1,DEEP_SIZE do
    model:push( ann.components.hyperplane{ input=isize, output=HSIZE },
                ann.components.actf.relu(),
                ann.components.dropout{ prob=0.5, random=prnd } )
    isize = nil
  end
  model:push( ann.components.hyperplane{ input=isize, output=NUM_CLASSES },
              ann.components.actf.log_softmax() )
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

local train_data = matrix.fromTabFilename("DATA/train_feats.noname.split.mat.gz")
local train_labels = matrix.fromTabFilename("DATA/train_labels.noname.split.mat.gz")
local val_data = matrix.fromTabFilename("DATA/val_feats.noname.split.mat.gz")
local val_labels = matrix.fromTabFilename("DATA/val_labels.noname.split.mat.gz")

print("# DATA SIZES", train_data:dim(1), train_data:dim(2),
      val_data:dim(1), val_data:dim(2))

local bagging_models = bagging(NUM_CLASSES, NUM_BAGS, MAX_FEATS, rnd,
                               train_data, train_labels,
                               val_data, val_labels,
                               train, predict_mlp)

local ce = ann.loss.multi_class_cross_entropy()
local val_p = predict_mlp(bagging_models, val_data)
local val_log_p = mop.log(val_p)
local val_in_ds,val_out_ds = create_ds(val_data, val_labels, NUM_CLASSES)
local tgt = val_out_ds:toMatrix()
ce:accum_loss(ce:compute_loss(val_log_p, val_out_ds:toMatrix()))
print("# VA LOSS", ce:get_accum_loss())

local cm = stats.confusion_matrix(NUM_CLASSES)
local _,val_cls = val_p:max(2)
cm:addData(dataset.matrix(val_cls:to_float()), dataset.matrix(val_labels))
cm:printConfusion()

write_submission("validation.mlp.csv", val_p)

-----------------------------------------------------------------------------

local test_data = matrix.fromTabFilename("DATA/test_feats.noname.split.mat.gz")
local test_p = predict_mlp(bagging_models, test_data)
print(test_p)

write_submission("result.mlp.csv", test_p)
