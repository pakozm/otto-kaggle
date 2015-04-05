local common = require "scripts.common"
local bagging    = common.bagging
local bootstrap  = common.bootstrap
local compute_center_scale = common.compute_center_scale
local create_ds  = common.create_ds
local load_CSV   = common.load_CSV
local predict    = common.predict
local preprocess = common.preprocess
local split      = common.split
local train_mlp  = common.train_mlp
local write_submission = common.write_submission
local posteriorKNN = knn.kdtree.posteriorKNN
local mop  = matrix.op
local rnd  = random(12394)
local wrnd = random(24825)
local srnd = random(52958)
local prnd = random(24925)
local use_all = tonumber(arg[1])
local NUM_BAGS = tonumber(arg[2] or 1)
local MAX_FEATS = tonumber(arg[3] or 9)
local NUM_CLASSES  = 9
local K = 1000

local function learn(data)
  local model = knn.kdtree(data:dim(2), rnd)
  model:push(data)
  model:build()
  return model
end

local isize
local function train_knn(train_data, train_labels, val_data, val_labels)
  isize = train_data:dim(2)
  return { train_data, train_labels }
end

local predict_knn = function(models, data)
  local func = function(model, data)
    local tr_data,labels = table.unpack(model)
    local kdt = learn(tr_data)
    local function index(id) return labels:get(id,1) end
    local outputs = matrix(data:dim(1), NUM_CLASSES):fill(-99)
    local p = parallel_foreach(4, data:dim(1),
                               function(i)
                                 local result = kdt:searchKNN(K, data[{i,':'}])
                                 local p = posteriorKNN(result, index)
                                 return p end)
    assert(#p == outputs:dim(1))
    for i=1,#p do
      for j,v in pairs(p[i]) do
        outputs[{i,j}] = v
      end
    end
    return outputs
  end
  local p = predict(models, data, func)
  return p
end

local all_train_data,all_train_labels = load_CSV("DATA/train.csv", false)
local all_train_data = preprocess(all_train_data, { add_nz=true, add_sum=true })
-- local center,scale = compute_center_scale(all_train_data)
local all_train_data,center,scale =
  stats.standardize(all_train_data, { center=center, scale=scale })
--local U,S,VT = stats.pca(all_train_data, { centered=true })
--local takeN,eigen_value,prob_mass=stats.pca.threshold(S, 0.99)
-- print("#",takeN,eigen_value,prob_mass)
--local all_train_data = stats.zca.whitening(all_train_data,U,S,eigen_value)
--
local train_data,val_data,train_labels,val_labels = split(rnd, 0.8,
                                                          all_train_data,
                                                          all_train_labels)

if use_all then
  train_data = all_train_data
  train_labels = all_train_labels
end

local val_in_ds,val_out_ds = create_ds(val_data, val_labels, NUM_CLASSES)
local bagging_models = bagging(NUM_CLASSES, NUM_BAGS, MAX_FEATS, rnd,
                               train_data, train_labels,
                               val_data, val_labels,
                               train_knn, predict_knn)

local ce = ann.loss.multi_class_cross_entropy()
local val_log_p = predict_knn(bagging_models, val_data)
local val_p = mop.exp(val_log_p)
local tgt = val_out_ds:toMatrix()
ce:accum_loss(ce:compute_loss(val_log_p, val_out_ds:toMatrix()))
print("# VA LOSS", ce:get_accum_loss())

local cm = stats.confusion_matrix(NUM_CLASSES)
local _,val_cls = val_p:max(2)
cm:addData(dataset.matrix(val_cls:to_float()), dataset.matrix(val_labels))
cm:printConfusion()

-----------------------------------------------------------------------------

--local model = knn.kdtree(isize, rnd)
--model:push(all_train_data)
--model:build()

local test_data,test_labels = load_CSV("DATA/test.csv", false)
local test_data = preprocess(test_data, { add_nz=true, add_sum=true })
local test_data = stats.standardize(test_data, { center=center, scale=scale })
--local test_data = stats.zca.whitening(test_data,U,S,eigen_value)

local test_p = predict_knn(bagging_models, test_data)
print(test_p)

write_submission("result_knn.csv", test_p)
