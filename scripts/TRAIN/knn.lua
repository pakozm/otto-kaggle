april_print_script_header(arg)

local common = require "scripts.common"
local bagging    = common.bagging
local create_ds  = common.create_ds
local predict    = common.predict
local split      = common.split
local write_submission = common.write_submission
local posteriorKNN = knn.kdtree.posteriorKNN
local mop  = matrix.op
local rnd  = random(12394)
local wrnd = random(24825)
local srnd = random(52958)
local prnd = random(24925)
local ID = assert(tonumber(arg[1]))
local NUM_BAGS = tonumber(arg[2] or 1)
local MAX_FEATS = tonumber(arg[3])
local feats_name = arg[4] or "std"
local NUM_CLASSES  = 9
local K = 1000

print("# num_bags max_feats feats")
print("#", NUM_BAGS, MAX_FEATS, feats_name)

local function learn(data)
  local model = knn.kdtree(data:dim(2), rnd)
  model:push(data)
  model:build()
  return model
end

local isize
local function train_knn(train_data, train_labels, val_data, val_labels)
  isize = train_data:dim(2)
  local N = math.round(0.1 * train_data:dim(1))
  local slice = { {1,N},':' }
  local train_data = train_data[slice]
  local train_labels = train_labels[slice]
  return { train_data, train_labels }
end

local predict_knn = function(models, data)
  collectgarbage("collect")
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

local train_data = matrix.fromTabFilename("DATA/train_feats.%s.split.mat.gz"%{feats_name})
local train_labels = matrix.fromTabFilename("DATA/train_labels.split.mat.gz")
local val_data = matrix.fromTabFilename("DATA/val_feats.%s.split.mat.gz"%{feats_name})
local val_labels = matrix.fromTabFilename("DATA/val_labels.split.mat.gz")

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

local test_data = matrix.fromTabFilename("DATA/test_feats.%s.split.mat.gz"%{feats_name})
local test_p = predict_knn(bagging_models, test_data)
print(test_p)

write_submission("result_knn.csv", test_p)
