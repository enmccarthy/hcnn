module Lib where

import Numeric.Matrix 
--INPUT--
    -- Number of input N
    -- the depth of input C
    -- Height of the input H
    -- Width of the input W
--HYPERPARAMETERS-- can I set default for this?
    -- Number of filters K
    -- Spatial Extent F
    -- Stride length S
    -- Zero Padding P
-- The spatial size of the output is given by
    -- (H-F+2P)/S+1*(W-F+2P)/S + 1
--hmm is there a better way to do this
-- can I set one variable at a time? 
-- possibly come back and make these default values 
-- see https://en.wikibooks.org/wiki/Haskell/More_on_datatypes

data Conv = Conv
    { d_X      :: Int
    , h_X      :: Int
    , w_X      :: Int
    }
data Hyperparameter = Hyperparameter 
    {
    , n_filter :: Int
    , h_filter :: Int
    , w_filter :: Int
    , stride   :: Int
    , padding  :: Int
    }
-- what do I call this
data Extra = Extra
    {
    , w        :: Matrix Double
    , b        :: Matrix Double
    , params   :: Matrix Double
    , h_out    :: Int
    , w_out    :: Int
    }
-- Forward Propagation

-- Create a matrix of size (h_filter*w_filer) by n_X * 
-- ((h_x-h_filter+2p)/stride +1)**2
-- layers
    -- Convolutional layer
    -- input layer 
    -- hidden layers
    -- fully connected layer
    -- pooling
        -- max pooling
-- activation functions
    -- relu
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
 

-- do I need a data type for weights/bias or just matrix 

--Mean squared error
-- old minus new divided by amount
mse :: [Double] -> [Double] -> Double
mse newy oldy =  ((sum [(i-j) | i <- oldy,
                               j <- newy])/ len)
    where len = fromIntegral(length newy)
--
--cross entropy loss
-- cel :: 