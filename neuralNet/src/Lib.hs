module Lib where

import qualified Numeric.LinearAlgebra as NL
import qualified Data.Vector as DV
import System.Random
import Data.Functor
import Control.Monad


data Layer = Input Int
    | Output Int
    | Hidden Int

--Loss Functions
--Mean squared error
mse :: [Float] -> [Float] -> Float
mse newy oldy =  ((sum [(i-j) | i <- oldy,
                               j <- newy])/ len)
    where len = fromIntegral(length newy)

-- following https://deepnotes.io/softmax-crossentropy
--softmax

softmax :: (DV.Vector Float) -> (DV.Vector Float)
softmax x =
    (DV.map (/(DV.sum toX)) toX)

    where toX = (DV.map exp (DV.map (subtract (DV.maximum x)) x))


--cross entropy loss
-- python code:
-- X is the output from fully connected layer (num_examples x num_classes)
-- y is labels (num_examples x 1)

cel :: DV.Vector Float -> DV.Vector Float -> Float
cel x y = (DV.sum (DV.map (\(pi,yi) -> -yi*(log pi)) zip1)) / (fromIntegral len)
    where zip1 = (DV.zipWith zipFloatTuple (softmax x) y)
          len  = (length y)

-- telling how I want them zipped, idk what will happen if they are different size
-- so if I have issues come back and look at this
zipFloatTuple :: Float -> Float -> (Float, Float)
zipFloatTuple a b = (a , b)

-- activation functions --
-- relu (boring)
relu :: Float -> Float
relu = max 0

type InputLayers = [Layer]
-- I need to keep track of the nodes in the previous layer
-- There is probably a better way to do this
type Model = [(Int, (IO Float, IO[(DV.Vector Float)]))]

type Error = [((DV.Vector Float), (DV.Vector Float))]

-- I make the assumption that input comes first and
-- output comes last but maybe I want to put a check somewhere
-- for that

-- takes a standard dev as input
-- grabbed from the internet
-- this is boxmuller
gauss :: Float -> IO Float
gauss scale = do
    x1 <- randomIO
    x2 <- randomIO
    return $ scale * sqrt (-2 * log x1) * cos (2 * pi * x2)

initModel :: Model -> InputLayers -> (Model, InputLayers)
initModel wholemod@((num, (b, w)):ms) (ls:lay) = case ls of
                (Input numIn) -> ([(numIn, (b,w))], lay)
                (Output numOut) -> ((reverse
                                      ((numOut,
                                        ((gauss (0.01)), (createW num numOut))):wholemod))
                                              , []) -- create the bias and weights for output layer
                (Hidden numHid) -> (((numHid,
                                      ((gauss (0.01)), (createW num numHid))):wholemod)
                                            , lay)

-- possibly a more efficient way to do this--
-- but wanted to make sure it gives different ran numbers
-- I need to check that it does
createW :: Int -> Int -> IO[(DV.Vector Float)]
createW num num2 = (replicateM num (createWHelp num2))

createWHelp :: Int -> IO(DV.Vector Float)
createWHelp num = do
    listRan <- (replicateM num (gauss (0.01)))
    return (DV.fromList listRan)


-- forward prop
--TODO set this up so it skips input layer
forwardprop :: Model -> ((DV.Vector Float), Error) -> IO((DV.Vector Float), Error)
forwardprop [] (vec, err) = return (vec, err)
forwardprop ((_, (b, w)):[]) (vec, err) = do
    weight <- w
    bias <- b
    let output = (softmax 
                    (DV.fromList
                    (map (+ bias)
                    (map (DV.foldl1 (+))
                    (map (DV.zipWith zipMult vec)
                    weight)))))
    return (output, ([(vec, output)] ++ err))

forwardprop ((_, (b, w)):ms) (vec, err) = do
    weight <- w
    bias <- b
    let output = (DV.fromList
                    (map relu
                    (map (+ bias)
                    (map (DV.foldl1 (+))
                    (map (DV.zipWith zipMult vec)
                        weight)))))
    (forwardprop ms (output, ((vec, output):err)))


zipMult :: Float -> Float -> Float
zipMult a b = (a * b)


-- backwards prop
--the derivative of cross entropy 
-- takes in the y vector and the output vector
derCE :: (DV.Vector Float) -> (DV.Vector Float) -> (DV.Vector Float)
derCE yvec outvec = (DV.zipWith zipAdd (DV.map (subtract 1) (DV.zipWith zipMult yvec outvec))
                                        (DV.zipWith zipMult (DV.map (\x -> 1.0 - x) yvec)
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
backwardsprop :: Model -> (DV.Vector Float) -> (DV.Vector Float) -> Error -> IO Model 
backwardsprop ((i, (b, w)):ms) out expout err@((beforeSM, afterSM):(beforeWeights, afterWeights):re) = do
        weight <- w
        let dce = (derCE expout out) --(vector)
        let dout = (derOut afterWeights) --(vector)
        let changeWeight = (DV.map (* 0.10) (DV.zipWith3 zipMult2 dce dout beforeWeights)) 
        let newWeight = return (map (DV.zipWith zipSub changeWeight) weight)
        otherMod <- (backprophelp ms dce dout weight err)
        -- new output layer weights
        return ([(i, (b, newWeight))] ++ otherMod)

-- layer to layer
-- derRelu * h1 outvalue * (derCE * derOut * (weights on 2nd layer output))

--input to layer
--derRelu * input value out * (previous relu * previous 3rd value * weight coming into layer)

backprophelp :: Model -> (DV.Vector Float) -> (DV.Vector Float) -> [(DV.Vector Float)] -> Error -> IO Model
backprophelp ((i, (b, w)):[]) d1 d2 weigh ((h2input, h2output):(h1input, h1output):re) = do
    currWeight <- w 
    let thirdValue = (map (DV.zipWith3 zipMult2 d1 d2) weigh) --[vector]
    let weightChange = (map (DV.map (* 0.10)) (map (DV.zipWith3 zipMult2 (derRelu h2input) h1output) thirdValue)) 
    let newWeight = return (zipWith zipVecSub weightChange currWeight) 
    return [(i, (b, newWeight))]
    
backprophelp ((i, (b, w)):ms) d1 d2 weigh ((h2input, h2output):(h1input, h1output):re) = do
    currWeight <- w 
    let thirdValue = (map (DV.zipWith3 zipMult2 d1 d2) weigh)
    let weightChange = (map (DV.map (* 0.10)) (map (DV.zipWith3 zipMult2 (derRelu h2input) h1output) thirdValue))
    let newWeight = return (zipWith zipVecSub weightChange currWeight) 
    otherMod <- (backprophelp1 ms (derRelu h2input) thirdValue currWeight ((h1input, h1output):re))
    return ([(i, (b, newWeight))] ++ otherMod)


backprophelp1 :: Model -> (DV.Vector Float) -> [(DV.Vector Float)] -> [(DV.Vector Float)] -> Error -> IO Model
backprophelp1 [] d1 d2 weigh _ = do
    return []
    
backprophelp1 ((i, (b, w)):[]) d1 d2 weigh _ = do
    return [(i, (b, w))]

backprophelp1 ((i, (b, w)):(i1, (b1, w1)):[]) d1 d2 weigh ((h2input, h2output):(h1input, h1output):re) = do
    currWeight <- w 
    let maper = (map (DV.zipWith zipMult d1) d2)
    let thirdValue = (zipWith zipVec maper weigh)
    let weightChange = (map (DV.map (* 0.10)) (map (DV.zipWith3 zipMult2 (derRelu h2input) h1output) thirdValue))
    let newWeight = return (zipWith zipVecSub weightChange currWeight)
    return [(i, (b, newWeight))]
    
backprophelp1 ((i, (b, w)):ms) d1 d2 weigh ((h2input, h2output):(h1input, h1output):re) = do
    currWeight <- w 
    let dRelu = (derRelu h2input)
    let maper = (map (DV.zipWith zipMult d1) d2)
    let thirdValue = (zipWith zipVec maper weigh)
    let weightChange = (map (DV.map (* 0.10)) (map (DV.zipWith3 zipMult2 dRelu h1output) thirdValue))
    let newWeight = return (zipWith zipVecSub weightChange currWeight)
    otherMod <- (backprophelp1 ms dRelu thirdValue currWeight ((h1input, h1output):re))
    return ([(i, (b, newWeight))] ++ otherMod)

--look to see where this is used and change it
zipVec :: (DV.Vector Float) -> (DV.Vector Float) -> (DV.Vector Float)
zipVec a b = DV.zipWith zipMult a b

zipVecSub :: (DV.Vector Float) -> (DV.Vector Float) -> (DV.Vector Float)
zipVecSub a b = DV.zipWith zipSub a b

zipMult2 :: Float -> Float -> Float -> Float
zipMult2 a b c= (a * b * c)

zipSub :: Float -> Float -> Float 
zipSub a b = (b - a)
