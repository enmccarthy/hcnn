module Lib where

import qualified Numeric.LinearAlgebra as NL
import qualified Data.Vector as DV
import System.Random 
import Data.Functor
import Control.Monad 

data Layer = Input Int
    | Output Int
    | Hidden Int 
-- weight ish
-- takes in an activation f weights bias and inputs
-- weight :: (Float -> Float) -> Vector Float -> Float -> Vector Float

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
type Model = [(Int, (IO (DV.Vector Float), IO[(DV.Vector Float)]))]

-- our initial weights and bias as well as number of layers


-- I make the assumption that input comes first and 
-- output comes last but maybe I want to put a check somewhere
-- for that

-- takes a standard dev as input 
-- grabbed from the internet
-- there are packages that have this implemented but 
-- I had problems downloading them 
-- this is boxmuller
gauss :: Float -> IO Float
gauss scale = do
    x1 <- randomIO
    x2 <- randomIO
    return $ scale * sqrt (-2 * log x1) * cos (2 * pi * x2)

initM :: Model -> InputLayers -> (Model, InputLayers)
initM wholemod@((num, (b, w)):ms) (ls:lay) = case ls of
                (Input numIn) -> ([(numIn, (b,w))], lay)
                (Output numOut) -> ((reverse ([(numOut, ((createB numOut), (createW num numOut)))] ++ wholemod)), []) -- create the bias and weights for output layer 
                (Hidden numHid) -> (([(numHid, ((createB numHid), (createW num numHid)))] ++ wholemod), lay)

--possibly a more efficient way than fromList
createB :: Int -> IO(DV.Vector Float)
createB num = do
            listRan <- (replicateM num (gauss (0.01)))
            return (DV.fromList listRan)
createW :: Int -> Int -> IO[(DV.Vector Float)]
createW num num2 = (replicateM num (createB num2))


-- forward prop
-- input the model
-- matrix multiplication summation and activations function
forwardprop :: Model -> (Dv.Vector Float) -> (DV.Vector Float)
forwardprop ((_, (b, w)):ms) vec =   

-- backwards prop
-- gradient descent

-- batch 
-- online learning