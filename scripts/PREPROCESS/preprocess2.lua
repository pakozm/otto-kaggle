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
local NAME = "raw2"
local INTERACTIONS = 0

local preprocess_conf = { add_nz=false,
                          add_max=false,
                          add_sum=false,
                          add_mean=false,
                          add_sd=false,
                          add_interactions=INTERACTIONS,
                          use_tf_idf=false,
                          ignore_counts=false }

local all_train_data,all_train_labels = load_CSV("DATA/train.csv")
local raw_train_data = all_train_data:clone()
local all_train_data,extra = preprocess(all_train_data, preprocess_conf)
-- local clusters = compute_clusters(all_train_data, all_train_labels, NUM_CLASSES)
-- local all_train_data = add_clusters_similarity(all_train_data, clusters)
-- local center,scale = compute_center_scale(all_train_data)
local all_train_data,center,scale =
  stats.standardize(all_train_data, { center=center, scale=scale })
--
local train_data,val_data,train_labels,val_labels,
raw_train_data,raw_val_data = split(rnd, 0.8,
                                    all_train_data,
                                    all_train_labels,
                                    raw_train_data)

collectgarbage("collect")

local test_data,test_labels = load_CSV("DATA/test.csv")
local test_data = preprocess(test_data, preprocess_conf, extra)
local test_data = stats.standardize(test_data, { center=center, scale=scale })

print("# DATA SIZES", train_data:dim(1), train_data:dim(2),
      val_data:dim(1), val_data:dim(2),
      test_data:dim(1), test_data:dim(2))

train_data:toTabFilename("DATA/train_feats.%s.split.mat.gz"%{NAME})
train_labels:toTabFilename("DATA/train_labels.split.mat.gz"%{NAME})
val_data:toTabFilename("DATA/val_feats.%s.split.mat.gz"%{NAME})
val_labels:toTabFilename("DATA/val_labels.split.mat.gz"%{NAME})
test_data:toTabFilename("DATA/test_feats.%s.split.mat.gz"%{NAME})
