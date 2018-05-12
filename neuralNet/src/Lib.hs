module Lib where

import qualified Numeric.LinearAlgebra as NL
import qualified Data.Vector as DV
import System.Random 
import Data.Functor 

data Layer = Input Int
    | Output Int
    | Hidden Int 
-- weight ish
-- takes in an activation f weights bias and inputs
weight :: (Float -> Float) -> Vector Float -> Float -> Vector Float

--Loss Functions
--Mean squared error
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
cel x y = (DV.sum (DV.map (\(pi,yi) -> -yi*(log pi)) zip1)) / (fromIntegral len)
    where zip1 = (DV.zipWith zipTheseV (softmax x) y)
          len  = (length y)

-- telling how I want them zipped, idk what will happen if they are different size
-- so if I have issues come back and look at this 
zipTheseV :: Float -> Float -> (Float, Float)
zipTheseV a b = (a , b)

-- activation functions --
-- TODO add more
-- relu (boring)
relu :: Float -> Float
relu = max 0

type InputLayers = [Layer] 
-- I need to keep track of the nodes in the previous layer
-- There is probably a better way to do this
type Model = [Int, (DV.Vector Float, [DL.Vector Float])]

type ModelState a = InputLayers -> (a, InputLayers)
-- our initial weights and bias as well as number of layers
-- if I extend this to conv then I can change it to a datatype 
-- and have it be a list of layers
-- following deal from class notes
-- it is possible that I don't need this output
-- I make the assumption that input comes first and 
-- output comes last but maybe I want to put a check somewhere
-- for that 

init :: Model -> InputLayers -> (Model, InputLayers)
init (num, (b, w)) ((Input numIn):xs) = ((numIn, (b,w)), xs)
init (num, (b, w)) ((Output Int):xs) = ()
init (num, (b, w))  ((Hidden Int):xs)
init' :: InputLayers -> ModelState Model
init'  = 

-- forward prop
-- backwards prop
-- gradient descent

-- batch 
-- online learning