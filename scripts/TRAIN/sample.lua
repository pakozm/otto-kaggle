local train_filename  = "train.csv"
local test_filename   = "test.csv"
local result_filename = "submission.csv"
local rnd = random(12394)
local bunch_size = 64 -- mini-batch size
local ratio = 0.8     -- 80% for training, 20% for validation
local topology = "93 inputs 200 relu dropout{prob=0.5,random=#1} 200 relu 9 log_softmax"
--
local cls_map = iterator.range(9):map(function(i) return "Class_"..i,i end):table()
local header_tbl = iterator.range(9):map(function(i) return i,"Class_"..i end):table()
table.insert(header_tbl, 1, "id")
--
local function load_train_data(filename)
  print("# Loading training data")
  local data = matrix.fromCSVFilename(filename, { header=true, map=cls_map })
  local N = data:dim(1)
  local shuf = matrixInt32(rnd:shuffle(N))
  local data = data:index(1,shuf) -- shuffled data
  
  local feats  = data[{ ':', {2,data:dim(2)-1} }]:log1p()
  local labels = data[{ ':', data:dim(2) }]
  
  local feats,center,scale = stats.standardize(feats)
  
  local M = math.ceil(ratio * N)
  local train_feats  = feats[{ {1,M}, ':' }]:clone()
  local train_labels = labels[{ {1,M}, ':' }]:clone()
  local val_feats    = feats[{ {M+1,N}, ':' }]:clone()
  local val_labels   = labels[{ {M+1,N}, ':' }]:clone()
  local feats,labels = nil,nil
  return train_feats,train_labels,val_feats,val_labels,center,scale
end

local function generate_trainer()
  print("# Generating trainer")
  local model = ann.mlp.all_all.generate(topology, { rnd })
  local trainer = trainable.supervised_trainer(model,
                                               ann.loss.multi_class_cross_entropy(),
                                               bunch_size,
                                               ann.optimizer.adadelta())
  trainer:build()
  trainer:randomize_weights{
    random = rnd,
    inf = -3,
    sup =  3,
    use_fanin=true,
    use_fanout=true,
  }
  return trainer
end

local function one_hot_ds(labels)
  return dataset.indexed( dataset.matrix(labels), { dataset.identity(9) } )
end

local train_feats, train_labels,
val_feats, val_labels, center, scale = load_train_data(train_filename)
collectgarbage("collect")

local trainer = generate_trainer()

local train_table = {
  input_dataset = dataset.matrix(train_feats),
  output_dataset = one_hot_ds(train_labels),
  shuffle = rnd,
}

local val_table = {
  input_dataset = dataset.matrix(val_feats),
  output_dataset = one_hot_ds(val_labels),
}

local pocket = trainable.train_holdout_validation{
  min_epochs = 4,
  max_epochs = 20,
  stopping_criterion = trainable.stopping_criteria.make_max_epochs_wo_imp_relative(1.5),
}

local function train_function()
  local tr = trainer:train_dataset(train_table)
  local va = trainer:validate_dataset(val_table)
  return trainer,tr,va
end

----------------------------------------------------------------------------

print("# Starting training")
while pocket:execute(train_function) do
  print(pocket:get_state_string())
end
local best = pocket:get_state_table().best

----------------------------------------------------------------------------

local test_feats = matrix.fromCSVFilename(test_filename, { header=true })
local test_feats = stats.standardize(test_feats[{':','2:'}]:log1p(),
                                     { center=center, scale=scale })
local output_ds = best:use_dataset{ input_dataset = dataset.matrix(test_feats) }
local i=0
local ids = matrix(test_feats:dim(1),1):map(function() i=i+1 return i end)
local results = output_ds:toMatrix():exp():clamp(1e-6, 1.0 - 1e-06)
local output = matrix.join(2, { ids, results })
output:toCSVFilename(result_filename, { header=header_tbl })
