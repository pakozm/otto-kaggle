local common = require "scripts.common"
local load_CSV = common.load_CSV
local mop = matrix.op
local bunch_size = 256

local data,labels = load_CSV("DATA/train.csv")
local data = common.preprocess(data, { add_nz=true, add_sum=true, add_sd=true })
local data,center,scale = stats.standardize(data, { center=center, scale=scale })
local clusters = common.compute_clusters(data, labels, 9)
-- local data = common.add_clusters_similarity(data, clusters)

local out_ds = dataset.indexed(dataset.matrix(labels),
                               { dataset.identity(9) })

local cors = matrix(data:dim(2),9)
for y=1,9 do
  collectgarbage("collect")
  local ym = labels:eq(y):to_float()labels:eq(y):to_float()
  for f=1,data:dim(2) do
    local c = stats.cor(data[{':',f}], ym, { centered=true })
    cors[{f,y}] = c:get(1,1)
  end
end

ImageIO.write(Image(cors:abs():adjust_range(0,1)), "cors_fy.png")

local cors = stats.cor(data)
ImageIO.write(Image(cors:abs():adjust_range(0,1)), "cors_ff.png")

local U,S,VT = stats.pca(data, { centered=true })
local takeN,eigen_value,prob_mass=stats.pca.threshold(S, 0.99)
print("#",takeN,eigen_value,prob_mass)
--local data = stats.pca.whitening(data,U,S,eigen_value)

for f=1,data:dim(2) do
  local col = data[{':',f}] -- {1,f}}]
  local model = ann.mlp.all_all.generate("%d input 9 log_softmax"%{col:dim(2)})
  local trainer = trainable.supervised_trainer(model,
                                               ann.loss.multi_class_cross_entropy(),
                                               bunch_size,
                                               ann.optimizer.adadelta())
  trainer:build()
  trainer:set_layerwise_option("w.*", "weight_decay", 0.01)
  local rnd = random(1234)
  trainer:randomize_weights{
    random = rnd,
    inf = -3,
    sup =  3,
    use_fanin = true,
    use_fanout = true,
  }
  for _,b in trainer:iterate_weights("b.*") do b:zeros() end
  local in_ds = dataset.matrix(col:clone())

  for i=1,200 do
    trainer:train_dataset{
      input_dataset = in_ds,
      output_dataset = out_ds,
      shuffle = rnd,
      replacement = bunch_size*10,
    }
  end
  print(f, trainer:validate_dataset{ input_dataset = in_ds,
                                     output_dataset = out_ds, })
end
