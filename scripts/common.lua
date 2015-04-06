local mop = matrix.op

local function compute_clusters(data, labels, NUM_CLASSES)
  local clusters = matrix(NUM_CLASSES, data:dim(2))
  for i=1,NUM_CLASSES do
    local idx = labels:eq(i)
    local cls_data = data:index(1, idx)
    local mu = stats.amean(cls_data, 1)
    clusters[i] = mu
  end
  clusters:toTabFilename("clusters.txt")
  return clusters
end

local function compute_row_norm2(data)
  local norm = matrix(data:dim(1), 1)
  for i,row in matrix.ext.iterate(data) do norm[i] = row:norm2() end
  return norm
end

local function add_clusters_similarity(data, clusters)
  local data_norm = compute_row_norm2(data)
  local clusters_norm = compute_row_norm2(clusters)
  local inv_norm = (data_norm * clusters_norm:t()):div(1)
  local similarity = (data * clusters:t()):cmul(inv_norm)
  return matrix.join(2, data, similarity)
end

local function write_submission(filename, test_p)
  local out = io.open(filename, "w")
  out:write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")
  out:write(iterator(matrix.ext.iterate(test_p)):
              select(2):
              map(matrix.."toTable"):
              map(bind(table.concat,nil,",")):
              enumerate():
              concat(",","\n"))
  out:write("\n")
  out:close()
end

local function compute_center_scale(data)
  local center = matrix(1, data:dim(2))
  local scale = matrix(1, data:dim(2))
  for i=1,data:dim(2) do
    local fb = data:select(2,i):gt(0)
    local f = data[{':',i}] -- data:index(1, fb)
    -- center[{1,i}] = stats.amean(f)
    scale[{1,i}],center[{1,i}] = stats.var(f)
  end
  center:zeros()
  scale:sqrt()
  return center,scale
end

local function create_ds(data, labels, NUM_CLASSES)
  local in_ds  = dataset.matrix(data)
  local out_ds = dataset.indexed(dataset.matrix(labels),
                                 { dataset.identity(NUM_CLASSES) })
  return in_ds,out_ds
end

local function train_mlp(trainer, max_epochs, train_tbl, val_tbl, use_all)
  local criterion = trainable.stopping_criteria.make_max_epochs_wo_imp_relative(2.0)
  local pocket = trainable.train_holdout_validation{ min_epochs=100,
                                                     max_epochs=use_all or max_epochs,
                                                     stopping_criterion = criterion }
  while pocket:execute(function()
      local tr = trainer:train_dataset(train_tbl)
      local va = trainer:validate_dataset(val_tbl)
      return trainer,tr,va
  end) do
    print(pocket:get_state_string(), trainer:norm2("w.*"), trainer:norm2("b.*"))
  end
  return pocket:get_state_table().best
end

local function tf_idf(data, idf)
  local tf  = data
  local gt0 = tf:gt(0):to_float()
  local N   = tf:dim(1)
  local idf = idf or ( N / gt0:sum(1) ):log()
  local result = matrix.ext.broadcast(tf.cmul, tf, idf)
  return result,idf
end

local function load_CSV(filename)
  print("# Loading", filename)
  local m,header = matrix.fromCSVFilename(filename, { header=true })
  local labels
  if #header == 95 then
    local f = io.open(filename)
    local labels_tbl = iterator(io.lines(filename)):
      tail():
      map(bind(string.tokenize, nil, ",")):
      field(95):
      map(bind(string.gsub, nil, "Class_", "")):
      select(1):
      map(tonumber):
      table()
    labels = matrix(#labels_tbl, 1, labels_tbl)
  end
  local m = m(':','2:94'):clone()
  return m,labels
end

local function preprocess(data,args,extra)
  local extra = extra or {}
  local new_cols,idf = {},nil
  local nz = data:gt(0):to_float():sum(2)
  if args.add_nz then
    table.insert(new_cols, nz)
  end
  if args.add_sum then
    table.insert(new_cols, data:sum(2))
  end
  if args.add_mean then
    table.insert(new_cols, data:sum(2) / nz)
  end
  if args.add_sd then
    table.insert(new_cols, (stats.var(data,2):scal(data:dim(2)-1)/nz):sqrt())
  end
  if args.add_interactions then
    local sum   = matrix.fromFilename("DATA/cors_sum.mat")
    local order = matrixInt32.fromFilename("DATA/cors_order.mat")
    local D = data:dim(2)
    assert(D*(D-1)/2.0 == sum:dim(1))
    assert(D*(D-1)/2.0 == order:dim(1))
    for a=1,args.add_interactions do
      local k = order[order:dim(1) - a + 1]
      local i = D - 2 - math.floor(math.sqrt(-8*k + 4*D*(D-1)-7)/2.0 - 0.5)
      local j = k + i + 1 - D*(D-1)/2 + (D-i)*((D-i)-1)/2
      local col = mop.cmul(data[{':',i}], data[{':',j}])
      table.insert(new_cols, col)
    end
  end
  if args.use_tf_idf then
    idf = extra.idf
    data,idf = tf_idf(data, idf)
  end
  if not args.use_tf_idf then
    data = mop.log1p(data)
  end
  if #new_cols > 0 then
    iterator(new_cols):apply(matrix.."log1p")
    data = matrix.join(2, data, table.unpack(new_cols))
  end
  return data,{ idf=idf }
end

local function split(rnd, p, ...)
  local t = table.pack(...)
  local shuf = matrixInt32(rnd:shuffle(t[1]:dim(1)))
  local t = iterator(t):map(function(m) return m:index(1, shuf) end):table()
  local result = {}
  local N1 = math.ceil(p * t[1]:dim(1))
  local N2 = t[1]:dim(1)
  for i=1,#t do
    table.insert(result, t[i][{ {1,N1}, ':' }]:clone())
    table.insert(result, t[i][{ {N1+1,N2}, ':' }]:clone())
  end
  return table.unpack(result)
end

local function predict(models, data, calculate)
  local p
  for i=1,#models do
    collectgarbage("collect")
    local filename,transform = table.unpack(models[i])
    local model = util.deserialize(filename)
    local current = calculate(model, transform(data))
    if not p then p = current else p:axpy(1.0, current) end
  end
  return p:scal(1/#models)
end

local function bootstrap(rnd, ...)
  local t = table.pack(...)
  local N = t[1]:dim(1)
  local boot = matrixInt32(N)
  for i=1,N do boot[i] = rnd:randInt(1,N) end
  local r = iterator(t):map(function(m) return m:index(1, boot) end):table()
  return table.unpack(r)
end

local function bagging(NUM_CLASSES, NUM_BAGS, MAX_FEATS, rnd,
                       train_data, train_labels,
                       val_data, val_labels,
                       train, predict)
  MAX_FEATS = MAX_FEATS or train_data:dim(2)
  local val_in_ds,val_out_ds = create_ds(val_data, val_labels, NUM_CLASSES)
  local val_out = val_out_ds:toMatrix()
  local models = setmetatable({}, {
      __gc = function(t) iterator(t):field(1):apply(os.remove) end
  })
  for i=1,NUM_BAGS do
    collectgarbage("collect")
    print("#",i,NUM_BAGS)
    local train_data,train_labels = train_data,train_labels
    if NUM_BAGS > 1 then
      train_data,train_labels = bootstrap(rnd, train_data, train_labels)
    end
    local transform = function(x) return x end
    if MAX_FEATS < train_data:dim(2) then
      local idx = matrixInt32(rnd:shuffle(train_data:dim(2)))[{ {1,MAX_FEATS} }]
      transform = function(x) return x:index(2,idx) end
      train_data = transform(train_data)
    end
    local mdl = train(train_data, train_labels, val_data, val_labels)
    local outname = os.tmpname()
    models[i] = { outname, transform }
    util.serialize(mdl, outname, "binary")
    --
    local ce = ann.loss.multi_class_cross_entropy()
    local val_log_p = predict({ models[i] }, val_data):log()
    ce:accum_loss(ce:compute_loss(val_log_p, val_out))
    print("# VA LOSS BAG", i, ce:get_accum_loss())
  end
  return models
end

-- exported functions
return {
  add_clusters_similarity = add_clusters_similarity,
  bagging = bagging,
  bootstrap = bootstrap,
  compute_center_scale = compute_center_scale,
  compute_clusters = compute_clusters,
  create_ds = create_ds,
  load_CSV = load_CSV,
  predict = predict,
  preprocess = preprocess,
  split = split,
  tf_idf = tf_idf,
  train_mlp = train_mlp,
  write_submission = write_submission,
}
