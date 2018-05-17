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
type Model = [(Int, (IO Float, IO[(DV.Vector Float)]))]

type Error = [((DV.Vector Float), (DV.Vector Float))]

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
                (Output numOut) -> ((reverse
                                      ([(numOut,
                                        ((gauss (0.01)), (createW num numOut)))]
                                            ++ wholemod))
                                              , []) -- create the bias and weights for output layer
                (Hidden numHid) -> (([(numHid,
                                      ((gauss (0.01)), (createW num numHid)))]
                                          ++ wholemod)
                                            , lay)


-- did this wrong so maybe come back to this to make it prettier
--possibly a more efficient way than fromList
createWHelp :: Int -> IO(DV.Vector Float)
createWHelp num = do
            listRan <- (replicateM num (gauss (0.01)))
            return (DV.fromList listRan)
createW :: Int -> Int -> IO[(DV.Vector Float)]
createW num num2 = (replicateM num (createWHelp num2))

-- do you use activation on the last layer (??) I think I need to modify this
-- last one maybe uses softmax
-- forward prop
forwardprop :: Model -> ((DV.Vector Float), Error) -> IO((DV.Vector Float), Error)
forwardprop [] (vec, err) = return (vec, err)
forwardprop ((_, (b, w)):[]) (vec, err) = do
    weight <- w
    bias <- b
    let output = (softmax 
                    (DV.fromList
                    (map (+ bias)
                    (map (DV.foldl1 (+))
                    (map (DV.zipWith zipTheseProp vec)
                    weight)))))
    return (output, ([(vec, output)] ++ err))

forwardprop ((_, (b, w)):ms) (vec, err) = do
    weight <- w
    bias <- b
    let output = (DV.fromList
                    (map relu
                    (map (+ bias)
                    (map (DV.foldl1 (+))
                    (map (DV.zipWith zipTheseProp vec)
                        weight)))))
    (forwardprop ms (output, ((vec, output):err)))


zipTheseProp :: Float -> Float -> Float
zipTheseProp a b = (a * b)


-- backwards prop
--the derivative of cross entropy 
-- takes in the y vector and the output vector
derCE :: (DV.Vector Float) -> (DV.Vector Float) -> (DV.Vector Float)
derCE yvec outvec = (DV.zipWith zipAdd (DV.map (subtract 1) (DV.zipWith zipTheseProp yvec outvec))
                                        (DV.zipWith zipTheseProp (DV.map (\x -> 1.0 - x) yvec)
                                                                    (DV.map (1.0/) (DV.map (\x -> 1.0 - x) outvec))))

zipAdd :: Float -> Float -> Float
zipAdd a b = (a + b)

-- dervative of each output with respect to their input 
-- takes in the input to the output nodes
derOut :: (DV.Vector Float) -> (DV.Vector Float)
derOut inp = (DV.map (/(inpSum^2))(DV.map (\x ->(exp x) *(inpSum - (exp x))) inp))
    where inpSum = (DV.sum (DV.map exp inp))

-- takes relu values
derRelu :: (DV.Vector Float) -> (DV.Vector Float) 
derRelu inp = (DV.map (\x -> if (x > 0) then 1 else 0) inp)


-- last layer to output-- 
-- derCE * derOut * layer before outvalue


-- backwards prop
-- TAKES A REVERSE MODEL/ERROR
-- takes the y of the output of the model and the expected y 
backwardsprop :: Model -> (DV.Vector Float) -> (DV.Vector Float) -> Error -> Model 
backwardsprop ((i, (b, w):ms) out expout ((beforeSM, afterSM):(beforeWeights, afterWeights):re) = do
    weight <- w
    bias <- b
    let dce = (derCE expout out)
    let dout = (derOut afterWeights)

    let changeWeight = (DV.map (* 0.10) (DV.zipWith zipTheseProp2 dce dout beforeWeights))
    let newWeight = (map (DV.zipWith zipTheseSub changeWeight) weight)
    -- new output layer weights
    [((i, (bias, newWeight))] ++ (backprophelp ms dce dout weight ((beforeWeights, afterWeights):re)))


-- layer to layer
-- derRelu * h1 outvalue * (derCE * derOut * (weights on 2nd layer output))

--input to layer
--derRelu * input value out * (previous relu * previous 3rd value * weight coming into layer)


backprophelp ((i, (b, w):ms) d1 d2 weigh ((h2input, h2output):(h1input, h1output):re) = do
    currWeight <- w 
    let dRelu = (derRelu h2input)
    let mult = DV.zipWith zipThereProp (DV.zipWith zipThereProp2 derRelu h1output d1) d2)
    let weightChange = (map (DV.zipWith zipThereProp mult) weigh)
    let newWeight = (DV.map (* 0.10) (DV.zipWith zipTheseSub weightChange currWeight)) 
    [((i, (b, newWeight))] ++ (backprophelp  

zipTheseProp2 :: Float -> Float -> Float -> Float
zipTheseProp2 a b c= (a * b * c)

zipTheseSub :: Float -> Float -> Float 
zipTheseSub a b = (b - a)