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
local use_all      = tonumber(arg[4])
local NUM_BAGS     = tonumber(arg[5] or 1)
local MAX_FEATS    = tonumber(arg[6])
local INTERACTIONS = tonumber(arg[7] or 0)

local max_epochs = 10000

local optimizer = "adadelta"
local options = {
  -- learning_rate = 0.0000,
  -- momentum = 0.9,
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
  trainer:set_layerwise_option("w.*", "weight_decay", 0.01)
  for name,value in ipairs(options) do
    trainer:set_option(name, value)
  end
  
  local train_in_ds,train_out_ds = create_ds(train_data, train_labels,
                                             NUM_CLASSES)
  local val_in_ds,val_out_ds = create_ds(val_data, val_labels, NUM_CLASSES)
  
  local train_in_ds = dataset.perturbation{ dataset  = train_in_ds,
                                            random   = prnd,
                                            variance = 0.2 }
  
  local best = train_mlp(trainer,
                         use_all or max_epochs,
                         { input_dataset = train_in_ds,
                           output_dataset = train_out_ds,
                           shuffle = srnd,
                           replacement = math.max(2560, bunch_size) },
                         { input_dataset = val_in_ds,
                           output_dataset = val_out_ds },
                         use_all)
  return best
end

local predict_mlp = function(models, data)
  local func = function(model, data)
    return model:calculate(data):exp()
  end
  local p = predict(models, data, func)
  return p
end

local preprocess_conf = { add_nz=true,
                          add_max=true,
                          add_sum=true,
                          add_mean=false,
                          add_sd=true,
                          add_interactions=INTERACTIONS,
                          use_tf_idf=false,
                          ignore_counts=false, }

local all_train_data,all_train_labels = load_CSV("DATA/train.csv")
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

local test_data,test_labels = load_CSV("DATA/test.csv")
local test_data = preprocess(test_data, preprocess_conf, extra)
local test_data = stats.standardize(test_data, { center=center, scale=scale })
--local test_data = add_clusters_similarity(test_data, clusters)
local test_data = stats.pca.whitening(test_data,U,S,eigen_value)
-- local test_data = test_data * U

local test_p = predict_mlp(bagging_models, test_data)
print(test_p)

write_submission("result.mlp.csv", test_p)
