module Lib where

import Numeric.Matrix 
    -- layers
        -- Convolutional layer
        -- input layer 
        -- hidden layers
        -- fully connected layer
-- activation functions
-- cost/loss function
    -- Mean squared error
    -- Cross Entropy loss
-- full batch vs stochastic gradient descent
-- weights / bias
-- input --output
-- user needs to specify dimensions or does a CNN not need that
type Dimensions = [Int]
data Layer = Input Dimensions
    | Output Dimensions

-- data Hyperparameters = 
    -- 

-- do I need a data type for weights/bias or just matrix 

--Mean squared error
-- old minus new divided by amount
mse :: [Int] -> [Int] -> Double
mse newy oldy =  (sum [(i-j) | i <- oldy,
                               j <- newy])/ len
    where len = length newy