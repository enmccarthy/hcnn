module Lib where

-- import Numeric.Matrix
import Numeric.LinearAlgebra 
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
-- maybe look at state / monads
data Conv = Conv
    { d_X      :: Int
    , h_X      :: Int
    , w_X      :: Int
    } deriving show
data Hyper = Hyper 
    { n_filter :: Int
    , h_filter :: Int
    , w_filter :: Int
    , stride   :: Int
    , padding  :: Int
    } deriving show
-- what do I call this
data Extra = Extra
    { w        :: Matrix Double
    , b        :: Matrix Double
    , params   :: Matrix Double
    , h_out    :: Int
    , w_out    :: Int
    } deriving show


-- set Extra once I have Hyperparameter and Conv

-- setExtra :: Conv -> Hyper -> Extra
-- setExtra (Conv d_X= dx) (Hyper nf hf wf str pad)  = (Extra w b para ho wo)
--     where w = 
--         b =
--         para =
--         ho =
--         wo = 

-- trying to create something equiv to np.random.randn(x,y,z,q)
-- might be reinventing the wheel buuut

ranRan3 :: Int -> Int -> Int -> [IO(Matrix Double)]
ranRan3 1 z q = let mat = (randn z q)
                in [mat]
ranRan3 y z q = let mat = (randn z q)
                  in [mat] ++ (ranRan3 (y-1) z q) 

ranRan4 :: Int -> Int -> Int -> Int -> [[IO(Matrix Double)]]
ranRan4 1 y z q = let her = (ranRan3 y z q)
                    in [her]
ranRan4 x y z q = let her = (ranRan3 y z q)
                    in [her] ++ (ranRan4 (x-1) y z q)

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