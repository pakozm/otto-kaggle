local mop = matrix.op

local function compute_interactions(data)
  local D = data:dim(2)
  local interactions = matrix(data:dim(1), D*(D-1)/2.0)
  local k=0
  print("# Interactions")
  for i=1,D-1 do
    collectgarbage("collect")
    for j=i+1,D do
      k=k+1
      interactions[{':',k}]:copy( data[{':',i}] ):cmul( data[{':',j}] )
    end
  end
  return interactions
end

local function load_ensemble_model_from_csv(filenames,tgt)
  local results = iterator(filenames):map(bind(matrix.fromCSVFilename, nil,
                                               { header=true })):
  map(function(m) return m[{':','2:'}]:clone() end):table()
  local C = results[1]:dim(2)
  if tgt then
    tgt = dataset.indexed(dataset.matrix(tgt),{dataset.identity(C)}):toMatrix()
  end
  local calculate = function(t,w)
    local results = t.results
    local w = w or matrix(#results):ones()
    assert(w:num_dim() == 1)
    w:scal(1/w:sum())
    local results = results
    local ensemble = results[1]:clone():scal(w[1])
    for i=2,#results do ensemble:axpy(w[i], results[i]) end
    return ensemble
  end
  local compute_loss = function(t,w)
    local tgt,results = t.tgt,t.results
    assert(tgt, "Unable to compute loss without targets")
    local log_ensemble = calculate(t,w):log()
    local ce = ann.loss.multi_class_cross_entropy()
    ce:accum_loss(ce:compute_loss(log_ensemble,tgt))
    local loss,var = ce:get_accum_loss()
    return loss,var
  end
  return { results = results, tgt = tgt,
           calculate = calculate, compute_loss = compute_loss, }
end

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

local function one_hot(train_labels, C)
  return dataset.indexed(dataset.matrix(train_labels),
                         { dataset.identity(C) }):toMatrix()
end

local function preprocess(data,args,extra)
  local extra = extra or {}
  local new_cols,idf = {},nil
  local nz = data:gt(0):to_float():sum(2):clamp(1.0,mathcore.limits.float.infinity())
  if args.add_nz then
    table.insert(new_cols, mop.log1p(nz))
  end
  if args.add_max then
    table.insert(new_cols, (data:max(2)):log1p())
  end
  if args.add_sum then
    table.insert(new_cols, data:sum(2):log1p())
  end
  if args.add_mean then
    table.insert(new_cols, (data:sum(2) / nz):log1p())
  end
  if args.add_sd then
    table.insert(new_cols, (stats.var(data,2):scal(data:dim(2)-1)/nz):sqrt():log1p())
  end
  if args.add_interactions and args.add_interactions>0 then
    local interactions = compute_interactions(data)
    interactions:log1p()
    extra.interactions = extra.interactions or {}
    local center,scale = extra.interactions.center,extra.interactions.scale
    local interactions,center,scale = stats.standardize(interactions,
                                                        { center=center,
                                                          scale=scale })
    local U,S,VT = extra.interactions.U,extra.interactions.S,nil
    if not U then
      print("# PCA")
      U,S,VT = stats.pca(interactions, { centered = true })
    end
    local takeN,eigen_value,prob_mass=stats.pca.threshold(S, 0.90)
    print("# PCA INTERACTIONS",takeN,eigen_value,prob_mass)
    local slice={1,args.add_interactions}
    interactions = stats.pca.whitening(interactions,
                                       U[{':',slice}],
                                       S[{slice,slice}],
                                       eigen_value)
    table.insert(new_cols, interactions)
    extra.interactions = {
      center = center,
      scale = scale,
      U = U,
      S = S,
    }
    if false then
      local sum   = matrix.fromFilename("DATA/coef_sum.mat")
      local order = matrixInt32.fromFilename("DATA/coef_order.mat")
      local D = data:dim(2)
      assert(D*(D-1)/2.0 == sum:dim(2))
      assert(D*(D-1)/2.0 == order:dim(1))
      for a=1,args.add_interactions do
        local n = D
        local k = order[order:dim(1) - a + 1] - 1
        local i = n - 2 - math.floor(math.sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5)
        local j = k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2
        i,j = i+1,j+1
        assert(i <= D and j <= D and i >= 0 and j >= 0 and i<j)
        local col = mop.cmul(data[{':',i}], data[{':',j}])
        table.insert(new_cols, col:log1p())
      end
    end
  end
  if args.use_tf_idf then
    idf = extra.idf
    data,idf = tf_idf(data, idf)
  end
  if not args.use_tf_idf then
    data = mop.log1p(data)
  end
  collectgarbage("collect")
  if #new_cols > 0 then
    if args.ignore_counts then
      data = matrix.join(2, table.unpack(new_cols))
    else
      data = matrix.join(2, data, table.unpack(new_cols))
    end
    new_cols = nil
  end
  collectgarbage("collect")
  return data,extra
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
  local STEP = 2048
  local p
  local N = data:dim(1)
  local sum = 0
  for i=1,#models do
    collectgarbage("collect")
    local filename,transform,w = table.unpack(models[i])
    w = w or 1/#models
    local model = util.deserialize(filename)
    for j=1,N,STEP do
      local range = {j, math.min(N,j+STEP-1)}
      local slice = data[{range,':'}]
      local current = calculate(model, transform(slice))
      if not p then p = matrix(N,current:dim(2)):zeros() end
      p[{range,':'}]:axpy(w, current)
    end
    sum = sum + w
  end
  p:scal(1/sum)
  return p
end

local function compute_loss_and_write_validation(NUM_CLASSES,
                                                 models, predict,
                                                 train_data, train_labels,
                                                 val_data, val_labels,
                                                 out_filename)
  local function compute_loss(data, labels)
    collectgarbage("collect")
    local ce = ann.loss.multi_class_cross_entropy()
    local p = predict(models, data)
    local log_p = mop.log(p)
    local tgt = one_hot(labels, NUM_CLASSES)
    ce:accum_loss(ce:compute_loss(log_p, tgt))
    local loss,var = ce:get_accum_loss()
    return loss,var,p
  end
  local tr_loss,tr_var = compute_loss(train_data, train_labels)
  local va_loss,va_var,val_p = compute_loss(val_data, val_labels)
  print("# TR LOSS", tr_loss, tr_var)
  print("# VA LOSS", va_loss, va_var)
  --
  local cm = stats.confusion_matrix(NUM_CLASSES)
  local _,val_cls = val_p:max(2)
  cm:addData(dataset.matrix(val_cls:to_float()), dataset.matrix(val_labels))
  cm:printConfusion()
  --
  write_submission(out_filename, val_p)
end


local function weighted_bootstrap(alpha, weights, rnd, ...)
  local t = table.pack(...)
  local alpha = alpha or 1
  local N = alpha*t[1]:dim(1)
  local weights = weights or matrix(N,1):fill(1/N)
  local dice = random.dice(weights:toTable())
  local boot = matrixInt32(N):map(function(x) return dice:thrown(rnd) end)
  local r = iterator(t):map(function(m) return m:index(1, boot) end):table()
  return table.unpack(r)
end

local function bootstrap(rnd, ...)
  local t = table.pack(...)
  local N = t[1]:dim(1)
  return weighted_bootstrap(nil, matrix(N):fill(1/N), rnd, ...)
end

-- http://web.stanford.edu/~hastie/Papers/samme.pdf (SAMME)
local function adaboost_SAMME_step(NUM_CLASSES, h_t, train_labels, ...)
  local N = train_labels:dim(1)
  local _,w = ...
  local w = w or matrix(N):fill(1/N)
  local _,I = ann.loss.zero_one():compute_loss(h_t, train_labels)
  -- weighted classification error loss
  local err = mop.cmul( I, w ):sum()/w:sum()
  -- local alpha = (math.log( (1 - err) / err) + math.log(NUM_CLASSES-1))
  local alpha = (math.log( (1 - err) / err) + math.log(NUM_CLASSES-1))
  if alpha < 0 then
    return
  end
  -- w:cmul( mop.exp( alpha * I ) )  
  w:cmul( mop.scal(I, 0.5/err) + mop.complement(I):scal(0.5/(1-err)) )
  w:scal( 1/w:sum() )
  return alpha,w
end

-- http://www.cis.upenn.edu/~mkearns/teaching/COLT/boostingexperiments.pdf (AdaBoost.M2)
-- http://www.en-trust.at/eibl/wp-content/uploads/sites/3/2013/08/Eibl01_ECML_digitboost.pdf
-- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.20.7205&rep=rep1&type=pdf
local function adaboost_M2_step(NUM_CLASSES, h_t, train_labels, ...)
  local N = train_labels:dim(1)
  local _,D,w,Y,B,q,W = ...
  local Y = Y or one_hot(train_labels, NUM_CLASSES)
  local B = B or mop.complement( Y )
  local D = D or matrix(N):fill(1/N)
  local w = w or matrix(N,NUM_CLASSES):fill(1/(N*(NUM_CLASSES-1))):cmul(B)
  local W = W or w:sum(2)
  local q = matrix.ext.broadcast(w.div, w, W, q):cmul(B)
  --
  local c_h_star = mop.cmul(h_t, Y):sum(2):complement()
  local h_g = mop.cmul(h_t, B)
  local ratio = c_h_star + mop.cmul( h_g, q ):sum(2):scal( 1/(NUM_CLASSES-1) )
  local pseudo_loss = 0.5 * mop.cmul(D, ratio):sum()
  if pseudo_loss > 0.5 then
    return
  end
  local alpha = 0.5 * math.log( (1 - pseudo_loss) / pseudo_loss )
  local ratio = matrix.ext.broadcast(math.add, c_h_star, h_g):scal( -alpha ):exp()
  w:cmul( ratio ):cmul( B )
  --
  local W = w:sum(2)
  D:copy(W):scal(1/W:sum())
  -- alpha value, samples distribution, besides internal state
  return alpha,D,w,Y,B,q,W
end

local function adaboost(method,
                        NUM_CLASSES, NUM_ITERS, rnd,
                        train_data, train_labels,
                        val_data, val_labels, train, predict)
  local alpha,weights,method_step
  local state = {}
  local Y = one_hot(train_labels, NUM_CLASSES)
  if method == "M2" then
    method_step = adaboost_M2_step
  elseif method == "SAMME" then
    method_step = adaboost_SAMME_step
  else
    error("Unknown adaboost method")
  end
  local models = {}
  local transform = function(x) return x end
  for i=1,NUM_ITERS do
    collectgarbage("collect")
    local model
    do
      local train_data,train_labels = train_data,train_labels
      if i > 1 then -- first iteration uses all training data
        train_data,train_labels = weighted_bootstrap(1.0,
                                                     weights,
                                                     rnd,
                                                     train_data,
                                                     train_labels)
      end
      model = train(train_data, train_labels, val_data, val_labels)
    end
    local outname = os.tmpname()
    models[i] = { outname, transform }
    util.serialize(model, outname, "binary")
    --
    -- FIXME: clamp if necessary?
    local h = predict({ models[i] }, train_data)
    state = table.pack(method_step(NUM_CLASSES, h, train_labels,
                                   table.unpack(state)))
    alpha,weights = state[1],state[2]
    if not alpha then break end
    models[i][3] = alpha
    --
    local p = predict(models, train_data):log()
    -- print(weights)
    print("# ALPHA", models[i][3])
    print("# TR", (ann.loss.multi_class_cross_entropy():compute_loss(p, Y)))
    weights:scal(1/weights:sum())
    -- weights:toTabFilename("jarl-%04d.mat"%{ i })
  end
  return models
end

local function logistic(x)
  return 1/(1 + math.exp(-x))
end

local function line_search(loss, h1, h2, Y)
  local log_h1 = mop.log(h1)
  local log_h2 = mop.log(h2)
  local xp = matrix{ 0.0 }
  local l
  local optimizer = ann.optimizer.simplex()
  optimizer:execute(function(w)
      local x = logistic(w.xp[1])
      local hat_log_y = mop.log( (1-x)*h1 + x*h2 )
      l = loss:compute_loss( hat_log_y, Y )
      return l end, { xp=xp })
  local x = logistic(xp[1])
  print("# LINE LAST", l, "::", x)
  return x
end

local function gradient_boosting(loss, learning_rate, LP,
                                 NUM_CLASSES, NUM_ITERS, rnd,
                                 train_data, train_labels,
                                 val_data, val_labels, train, predict,
                                 val_filename)
  local state = {}
  local N = train_data:dim(1)
  local Y = one_hot(train_labels, NUM_CLASSES)
  local models = {}
  local transform = function(x) return x end
  for i=1,NUM_ITERS do
    collectgarbage("collect")
    local model
    do
      local train_data,train_labels = train_data,train_labels
      --if i > 1 then -- first iteration uses all training data
      train_data,train_labels = weighted_bootstrap(1.0,
                                                   weights,
                                                   rnd,
                                                   train_data,
                                                   train_labels)
      --end
      model = train(train_data, train_labels, val_data, val_labels)
    end
    local outname = os.tmpname()
    model_tbl = { outname, transform }
    util.serialize(model, outname, "binary")
    -- compute model combination coefficient
    local alpha = 1.0
    if #models > 0 then
      -- FIXME: clamp if necessary?
      -- FIXME: this can be taken from previous iteration
      local old_h = predict(models, train_data)
      local h = predict({ model_tbl }, train_data)
      alpha = line_search(loss, old_h, h, Y)
      for k=1,#models do models[k][3] = models[k][3] * (1.0-alpha) end
    end
    models[i] = model_tbl
    models[i][3] = learning_rate * alpha
    -- update weights
    local h = predict(models, train_data)
    local log_h = mop.log( h )
    local err   = loss:compute_loss(log_h, Y)
    local abs_grads = loss:gradient(log_h, Y):abs()
    weights = weights or matrix(N, 1)
    abs_grads:sum( 2, weights )
    weights:pow( 1/LP )
    weights:scal( 1/weights:sum() )
    --
    print("# TR", (ann.loss.multi_class_cross_entropy():compute_loss(log_h, Y)))
    print("# ALPHA", models[i][3])
  end
  compute_loss_and_write_validation(NUM_CLASSES,
                                    models, predict,
                                    train_data, train_labels,
                                    val_data, val_labels,
                                    val_filename)

  return models
end

local function bagging(NUM_CLASSES, NUM_BAGS, MAX_FEATS, rnd,
                       train_data, train_labels,
                       val_data, val_labels,
                       train, predict, val_filename)
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
    end
    local mdl = train(transform(train_data), train_labels,
                      transform(val_data), val_labels)
    local outname = os.tmpname()
    models[i] = { outname, transform }
    util.serialize(mdl, outname, "binary")
    --
    local ce = ann.loss.multi_class_cross_entropy()
    local val_log_p = predict({ models[i] }, val_data):log()
    ce:accum_loss(ce:compute_loss(val_log_p, val_out))
    print("# VA LOSS BAG", i, ce:get_accum_loss())
  end
  compute_loss_and_write_validation(NUM_CLASSES,
                                    models, predict,
                                    train_data, train_labels,
                                    val_data, val_labels,
                                    val_filename)
  return models
end

-- exported functions
return {
  adaboost = adaboost,
  add_clusters_similarity = add_clusters_similarity,
  bagging = bagging,
  bootstrap = bootstrap,
  compute_center_scale = compute_center_scale,
  compute_clusters = compute_clusters,
  compute_interactions = compute_interactions,
  create_ds = create_ds,
  gradient_boosting = gradient_boosting,
  load_CSV = load_CSV,
  load_ensemble_model_from_csv = load_ensemble_model_from_csv,
  predict = predict,
  preprocess = preprocess,
  split = split,
  tf_idf = tf_idf,
  train_mlp = train_mlp,
  write_submission = write_submission,
}
