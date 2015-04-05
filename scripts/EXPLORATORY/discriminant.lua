local common = require "scripts.common"
local load_CSV = common.load_CSV
local mop = matrix.op
local bunch_size = 256

local data,labels = load_CSV("DATA/train.csv")
local N = data:dim(1)
local D = data:dim(2)
local interactions = matrix(N, D*D/2.0)
local k=1
for i=1,D-1 do
  collectgarbage("collect")
  print("#",i)
  for j=i+1,D do
    interactions[{':',k}]:copy( data[{':',i}] ):cmul( data[{':',j}] )
    k=k+1
  end
end
local data = matrix.join(2, data, interactions)
data:log1p()
-- local data = common.preprocess(data)
local data,center,scale = stats.standardize(data, { center=center, scale=scale })
-- local clusters = common.compute_clusters(data, labels, 9)
-- local data = common.add_clusters_similarity(data, clusters)
print("# Computing correlations")
local cors = matrix(data:dim(2),9)
for y=1,9 do
  collectgarbage("collect")
  local ym = labels:eq(y):to_float()labels:eq(y):to_float()
  for f=1,data:dim(2) do
    local c = stats.cor(data[{':',f}], ym, { centered=true })
    cors[{f,y}] = c:get(1,1)
  end
end

local cors_sum = mop.abs(cors):sum(2)
local cors_rank = cors_sum:order_rank()
cors_rank:toFilename("cors_rank.mat", "ascii")
cors_sum:toFilename("cors_sum.mat", "ascii")

ImageIO.write(Image(cors:abs():adjust_range(0,1)), "cors_fy.png")

os.exit(0)

local out_ds = dataset.indexed(dataset.matrix(labels),
                               { dataset.identity(9) })


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
