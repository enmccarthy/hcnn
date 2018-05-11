module Lib where

import qualified Numeric.LinearAlgebra as NL
import qualified Data.Vector as DV
    --Activation Func

--Loss Functions
--Mean squared error
-- old minus new divided by amount
mse :: [Float] -> [Float] -> Float
mse newy oldy =  ((sum [(i-j) | i <- oldy,
                               j <- newy])/ len)
    where len = fromIntegral(length newy)

-- following https://deepnotes.io/softmax-crossentropy
--softmax
-- python code for it:
-- def softmax(X):
--     exps = np.exp(X-np.max(X))
--     return exps / np.sum(exps)
softmax :: (DV.Vector Float) -> (DV.Vector Float)
softmax x = 
    (DV.map (/(DV.sum toX)) toX)

    where toX = (DV.map exp (DV.map (subtract (DV.maximum x)) x))
 
                  
--cross entropy loss
-- python code:
-- X is the output from fully connected layer (num_examples x num_classes)
-- y is labels (num_examples x 1)

-- def cross_entropy(X,y):
    -- m = y.shape[0]
    -- p = softmax(X)
    -- log_likelihood = -np.log(p[range(m),y])
    -- loss = np.sum(log_likelihood) / m
    -- return loss

cel :: DV.Vector Float -> DV.Vector Float -> Float
cel x y = (DV.sum (DV.map (\(pi,yi) -> yi*(log pi)) zip1)) / (fromIntegral len)
    where zip1 = (DV.zipWith zipTheseV (softmax x) y)
          len    = (length y)
          
zipTheseV :: Float -> Float -> (Float, Float)
zipTheseV a b = (a , b)

type Dimensions = [Int]
data Layer = Input Dimensions
    | Output Dimensions
    | Hidden 