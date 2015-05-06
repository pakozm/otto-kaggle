local common = require "scripts.common"
local add_clusters_similarity = common.add_clusters_similarity
local compute_center_scale = common.compute_center_scale
local compute_clusters     = common.compute_clusters
local load_CSV   = common.load_CSV
local preprocess = common.preprocess
local split      = common.split
local mop  = matrix.op
local rnd  = random(12394)
local NUM_CLASSES  = 9
local NAME = arg[1] or "std"
local INTERACTIONS = tonumber(arg[2] or 0)

local preprocess_conf = { add_nz=true,
                          add_max=true,
                          add_sum=true,
                          add_mean=false,
                          add_sd=true,
                          add_interactions=INTERACTIONS,
                          use_tf_idf=false,
                          ignore_counts=false, }

local all_train_data,all_train_labels = load_CSV("DATA/train.csv")
local raw_train_data = all_train_data:clone()
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
local train_data,val_data,train_labels,val_labels,
raw_train_data,raw_val_data = split(rnd, 0.8,
                                    all_train_data,
                                    all_train_labels,
                                    raw_train_data)

raw_train_data:toTabFilename("DATA/train_feats.raw.split.mat.gz")
raw_val_data:toTabFilename("DATA/val_feats.raw.split.mat.gz")
raw_train_data=nil
raw_val_data=nil
collectgarbage("collect")

local test_data,test_labels = load_CSV("DATA/test.csv")
test_data:toTabFilename("DATA/test_feats.raw.split.mat.gz"%{NAME})
local test_data = preprocess(test_data, preprocess_conf, extra)
local test_data = stats.standardize(test_data, { center=center, scale=scale })
--local test_data = add_clusters_similarity(test_data, clusters)
local test_data = stats.pca.whitening(test_data,U,S,eigen_value)
-- local test_data = test_data * U

print("# DATA SIZES", train_data:dim(1), train_data:dim(2),
      val_data:dim(1), val_data:dim(2),
      test_data:dim(1), test_data:dim(2))

train_data:toTabFilename("DATA/train_feats.%s.split.mat.gz"%{NAME})
train_labels:toTabFilename("DATA/train_labels.split.mat.gz"%{NAME})
val_data:toTabFilename("DATA/val_feats.%s.split.mat.gz"%{NAME})
val_labels:toTabFilename("DATA/val_labels.split.mat.gz"%{NAME})
test_data:toTabFilename("DATA/test_feats.%s.split.mat.gz"%{NAME})
