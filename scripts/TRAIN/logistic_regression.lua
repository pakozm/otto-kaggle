april_print_script_header(arg)

local stdml  = require "stdml"
local common = require "scripts.common"
local bagging    = common.bagging
local create_ds  = common.create_ds
local predict    = common.predict
local write_submission = common.write_submission
local mop  = matrix.op
local rnd  = random(12394)
local wrnd = random(24825)
local srnd = random(52958)
local prnd = random(24925)
local NUM_CLASSES  = 9
local ID = assert(tonumber(arg[1]))
local bunch_size   = tonumber(arg[2] or 512)
local NUM_BAGS     = tonumber(arg[3] or 1000)
local MAX_FEATS    = tonumber(arg[4])

local optimizer = "adadelta"
local options = {
  -- learning_rate = 1.0,
  -- momentum = 0.9,
}

print("# bunch_size, num_bags, max_feats")
print("#", bunch_size, NUM_BAGS, MAX_FEATS)
print("# optimizer", optimizer)
print("# options", iterator(pairs(options)):concat("="," "))

local bagging_iteration=0
local function train(train_data, train_labels, val_data, val_labels)
  local HSIZE = HSIZE
  bagging_iteration = bagging_iteration + 1
  local isize = train_data:dim(2)
  local model = stdml.linear_model.logistic_regression{
    l2 = 0.00,
    shuffle = srnd,
    verbose = false,
    bunch_size = bunch_size,
    method = optimizer,
    options = options,
  }
  model:fit(train_data, train_labels, val_data, val_labels)
  return model
end

local predict_mlp = function(models, data)
  local func = function(model, data)
    return model:predict_proba(data)
  end
  local p = predict(models, data, func)
  return p
end

local train_data = matrix.fromTabFilename("DATA/train_feats.int600.split.mat.gz")
local train_labels = matrix.fromTabFilename("DATA/train_labels.split.mat.gz")
local val_data = matrix.fromTabFilename("DATA/val_feats.int600.split.mat.gz")
local val_labels = matrix.fromTabFilename("DATA/val_labels.split.mat.gz")

print("# DATA SIZES", train_data:dim(1), train_data:dim(2),
      val_data:dim(1), val_data:dim(2))

local bagging_models = bagging(NUM_CLASSES, NUM_BAGS, MAX_FEATS, rnd,
                               train_data, train_labels,
                               val_data, val_labels,
                               train, predict_mlp,
                               "ID_%03d.validation.lr.csv"%{ID})

-----------------------------------------------------------------------------

local test_data = matrix.fromTabFilename("DATA/test_feats.int600.split.mat.gz")
local test_p = predict_mlp(bagging_models, test_data)
print(test_p)

write_submission("ID_%03d.test.lr.csv"%{ID}, test_p)
