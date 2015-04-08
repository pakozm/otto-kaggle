local stdml  = require "stdml"
local common = require "scripts.common"
local adaboost   = common.adaboost
local create_ds  = common.create_ds
local gradient_boosting = common.gradient_boosting
local predict    = common.predict
local write_submission = common.write_submission
local mop  = matrix.op
local rnd  = random(12394)
local wrnd = random(24825)
local srnd = random(52958)
local prnd = random(24925)
local NUM_CLASSES  = 9
local bunch_size   = tonumber(arg[1] or 512)
local NUM_BAGS     = tonumber(arg[2] or 10)

local ver="int600"

local optimizer = "adadelta"
local options = {
  --learning_rate = 0.01,
  --momentum = 0.2,
}

local function train(train_data, train_labels, val_data, val_labels)
  local isize = train_data:dim(2)
  local model = stdml.linear_model.logistic_regression{
    l2 = 0.01,
    shuffle = srnd,
    verbose = true,
    bunch_size = bunch_size,
    method = optimizer,
    options = options,
    num_classes = NUM_CLASSES,
  }
  model:fit(train_data, train_labels, val_data, val_labels)
  local Y = dataset.indexed(dataset.matrix(val_labels),
                            { dataset.identity(NUM_CLASSES) }):toMatrix()
  print("# VAL", (ann.loss.multi_class_cross_entropy():
                    compute_loss(model:predict_log_proba(val_data),Y)))
  return model
end

local predict_mlp = function(models, data)
  local func = function(model, data)
    return model:predict_proba(data)
  end
  local p = predict(models, data, func)
  return p
end

local train_data = matrix.fromTabFilename("DATA/train_feats.%s.split.mat.gz"%{ver})
local train_labels = matrix.fromTabFilename("DATA/train_labels.%s.split.mat.gz"%{ver})
local val_data = matrix.fromTabFilename("DATA/val_feats.%s.split.mat.gz"%{ver})
local val_labels = matrix.fromTabFilename("DATA/val_labels.%s.split.mat.gz"%{ver})

print("# DATA SIZES", train_data:dim(1), train_data:dim(2),
      val_data:dim(1), val_data:dim(2))

--local models = adaboost(method,
--                        NUM_CLASSES, NUM_BAGS, rnd,
--                        train_data, train_labels,
--                        val_data, val_labels,
--                        train, predict_mlp)

local models = gradient_boosting(ann.loss.multi_class_cross_entropy(),
                                 1.0, 2.0,
                                 NUM_CLASSES, NUM_BAGS, rnd,
                                 train_data, train_labels,
                                 val_data, val_labels,
                                 train, predict_mlp)

local ce = ann.loss.multi_class_cross_entropy()
local val_p = predict_mlp(models, val_data)
local val_log_p = mop.log(val_p)
local val_in_ds,val_out_ds = create_ds(val_data, val_labels, NUM_CLASSES)
local tgt = val_out_ds:toMatrix()
ce:accum_loss(ce:compute_loss(val_log_p, val_out_ds:toMatrix()))
print("# VA LOSS", ce:get_accum_loss())

local cm = stats.confusion_matrix(NUM_CLASSES)
local _,val_cls = val_p:max(2)
cm:addData(dataset.matrix(val_cls:to_float()), dataset.matrix(val_labels))
cm:printConfusion()

write_submission("validation.blr.csv", val_p)

-----------------------------------------------------------------------------

local test_data = matrix.fromTabFilename("DATA/test_feats.%s.split.mat.gz"%{ver})
local test_p = predict_mlp(models, test_data)
print(test_p)

write_submission("result.blr.csv", test_p)
