local stdml = require "stdml"
local common = require "scripts.common"
local load_CSV = common.load_CSV
local mop = matrix.op
local bunch_size = 256

local data,labels = load_CSV("DATA/train.csv")
local N = data:dim(1)
local D = data:dim(2)
local interactions = matrix(N, D*(D-1)/2.0)
local k=1
for i=1,D-1 do
  collectgarbage("collect")
  print("#",i)
  for j=i+1,D do
    interactions[{':',k}]:copy( data[{':',i}] ):cmul( data[{':',j}] )
    k=k+1
  end
end
local data = interactions
data:log1p()
local data,center,scale = stats.standardize(data, { center=center, scale=scale })

print("# Training logistic regression model")
local Y = matrix(9,9):zeros():diag(1):index(1,matrixInt32(labels:toTable()))
local model = stdml.linear_model.logistic_regression{ fit_intercept = false,
                                                      verbose = true,
                                                      l2 = 0.01,
                                                      min_epochs = 20 }
model:fit(data, labels)
local coef = model.coef_
coef:toFilename("DATA/coef.mat", "ascii")
local coef_sum = mop.abs(coef):sum(1)
local coef_order = coef_sum:order()
coef_sum:toFilename("DATA/coef_sum.mat", "ascii")
coef_order:toFilename("DATA/coef_order.mat", "ascii")

print("# Computing correlations")
local cors = matrix(data:dim(2),9)
-- local aux = { data }
for y=1,9 do
  collectgarbage("collect")
  local ym = labels:eq(y):to_float()
  for f=1,data:dim(2) do
    local c = stats.cor(data[{':',f}], ym, { centered=true }):get(1,1)
    cors[{f,y}] = c
  end
  -- table.insert(aux, ym)
end

-- local aux = matrix.join(2, aux)
-- aux:toTabFilename("interactions.dat")

local cors_sum = mop.abs(cors):sum(2)
local cors_order = cors_sum:order()
cors_order:toFilename("DATA/cors_order.mat", "ascii")
cors_sum:toFilename("DATA/cors_sum.mat", "ascii")
ImageIO.write(Image(cors:abs():adjust_range(0,1)), "interactions_fy.png")
