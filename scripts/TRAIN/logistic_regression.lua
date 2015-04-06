local stdml  = require "stdml"
local common = require "scripts.common"
local add_clusters_similarity = common.add_clusters_similarity
local bagging    = common.bagging
local bootstrap  = common.bootstrap
local compute_center_scale = common.compute_center_scale
local compute_clusters     = common.compute_clusters
local create_ds  = common.create_ds
local load_CSV   = common.load_CSV
local predict    = common.predict
local preprocess = common.preprocess
local split      = common.split
local tf_idf     = common.tf_idf
local write_submission = common.write_submission
local mop  = matrix.op
local rnd  = random(12394)
local wrnd = random(24825)
local srnd = random(52958)
local prnd = random(24925)
local NUM_CLASSES  = 9
local bunch_size   = tonumber(arg[1] or 512)
local use_all      = tonumber(arg[2])
local NUM_BAGS     = tonumber(arg[3] or 1)
local MAX_FEATS    = tonumber(arg[4])
local INTERACTIONS = tonumber(arg[5] or 100)

local optimizer = "adadelta"
local options = {
  -- learning_rate = 1.0,
  -- momentum = 0.9,
}

local bagging_iteration=0
local function train(train_data, train_labels, val_data, val_labels)
  local HSIZE = HSIZE
  bagging_iteration = bagging_iteration + 1
  local isize = train_data:dim(2)
  local model = stdml.linear_model.logistic_regression{
    l2 = 0.01,
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

local preprocess_conf = { add_nz=true,
                          add_sum=true,
                          add_mean=false,
                          add_sd=true,
                          add_interactions=INTERACTIONS,
                          use_tf_idf=false }

local all_train_data,all_train_labels = load_CSV("DATA/train.csv", false)
local all_train_data,extra = preprocess(all_train_data, preprocess_conf)
-- local clusters = compute_clusters(all_train_data, all_train_labels, NUM_CLASSES)
-- local all_train_data = add_clusters_similarity(all_train_data, clusters)
-- local center,scale = compute_center_scale(all_train_data)
local all_train_data,center,scale =
  stats.standardize(all_train_data, { center=center, scale=scale })
local U,S,VT = stats.pca(all_train_data, { centered=true })
local takeN,eigen_value,prob_mass=stats.pca.threshold(S, 0.99)
print("#",takeN,eigen_value,prob_mass)
local all_train_data = stats.pca.whitening(all_train_data,U,S,eigen_value)
-- local all_train_data = all_train_data * U
--
local train_data,val_data,train_labels,val_labels = split(rnd, 0.8,
                                                          all_train_data,
                                                          all_train_labels)
if use_all then
  train_data = all_train_data
  train_labels = all_train_labels
end

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

-----------------------------------------------------------------------------

local test_data,test_labels = load_CSV("DATA/test.csv", false)
local test_data = preprocess(test_data, preprocess_conf, extra)
local test_data = stats.standardize(test_data, { center=center, scale=scale })
--local test_data = add_clusters_similarity(test_data, clusters)
local test_data = stats.pca.whitening(test_data,U,S,eigen_value)
-- local test_data = test_data * U

local test_p = predict_mlp(bagging_models, test_data)
print(test_p)

write_submission("result.lr.csv", test_p)
