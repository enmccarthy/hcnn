module Lib where

import qualified Numeric.LinearAlgebra as NL
-- import qualified Data.Vector as DV
import System.Random
import Data.Functor
import Control.Monad


data Layer = Input Int
    | Output Int
    | Hidden Int

--Loss Functions
--Mean squared error
mse :: [Float] -> [Float] -> Float
mse newy oldy =  (((sum [(i-j) | i <- oldy,
                               j <- newy])**2)/ len)
    where len = fromIntegral(length newy)

-- following https://deepnotes.io/softmax-crossentropy
--softmax

softmax :: [Float] ->  [Float]
softmax x =
    (map (/(sum toX)) toX)

    where toX = (map exp (map (subtract (maximum x)) x))


--cross entropy loss
-- python code:
-- X is the output from fully connected layer (num_examples x num_classes)
-- y is labels (num_examples x 1)
--CHECK THIS becuase I think I have confused myself with x/y and where softmax
-- should be applied, I actually think on y now
cel :: [[Float]] -> [Float] -> [Float]
cel x y = map (\(a,b) -> a/ b )
  (zipWith zipIntTuple
    (map (sum)
      (map (map (\(yi,xi) -> -yi*(log xi))) zip1))
    len)
    where zip1 = (map (zipWith zipFloatTuple y) (map softmax x))
          len  = (map (length) x)

-- telling how I want them zipped, idk what will happen if they are different size
-- so if I have issues come back and look at this
zipFloatTuple :: Float -> Float -> (Float, Float)
zipFloatTuple a b = (a , b)

zipIntTuple :: Float -> Int -> (Float, Float)
zipIntTuple a b = (a , (fromIntegral b))
-- activation functions --
-- relu (boring)
relu :: Float -> Float
relu = max 0

type InputLayers = [Layer]
-- I need to keep track of the nodes in the previous layer
-- There is probably a better way to do this
type Model = [(Int, (IO Float, IO[[Float]]))]

type Error = [([Float], [Float])]

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
initModel wholemod@((num, (b, w)):ms) (ls:lay) =
  case ls of
    (Input numIn)   -> ([(numIn, (b,w))], lay)
    (Output numOut) -> ((reverse
                          ((numOut,
                            ((gauss (0.01)), (createW numOut num))):wholemod))
                                , []) -- create the bias and weights for output layer
    (Hidden numHid) -> (((numHid,
                          ((gauss (0.01)), (createW numHid num))):wholemod)
                                , lay)

-- possibly a more efficient way to do this--
-- but wanted to make sure it gives different ran numbers
-- I need to check that it does
createW :: Int -> Int -> IO[[Float]]
createW num num2 = (replicateM num (createWHelp num2))

createWHelp :: Int -> IO [Float]
createWHelp num = do
    -- listRan <- (replicateM num (gauss (0.01)))
    (replicateM num (gauss (0.01)))


-- forward prop

forwardprop :: Model -> ([Float], Error)
                        -> IO([Float], Error)
forwardprop [] (vec, err) = return (vec, err)
forwardprop ((_, (b, w)):[]) (vec, err) = do
    weight <- w
    bias <- b
    let output = (softmax 
                    (map (+ bias)
                    (map (foldl1 (+))
                    (multWeight (zipMultWeight vec
                    weight)))))
    return (output, ([(vec, output)] ++ err))

forwardprop ((_, (b, w)):ms) (vec, err) = do
    weight <- w
    bias <- b
    let output = (map relu
                    (map (+ bias)
                    (map (foldl1 (+))
                    (multWeight (zipMultWeight vec 
                        weight)))))
    (forwardprop ms (output, ((vec, output):err)))

zipMultWeight :: [Float] -> [[Float]] -> [([Float],[Float])]
zipMultWeight a [] = []
zipMultWeight a (b:bs) = [(a,b)] ++ (zipMultWeight a bs)

multWeight [] = []
multWeight ((a,b):rs) = [(zipWith zipMult a b)] ++ (multWeight rs)

zipMult :: Float -> Float -> Float
zipMult a b = (a * b)


-- backwards prop
--the derivative of cross entropy
-- takes in the y vector and the output vector
derCE :: [Float] -> [Float] -> [Float]
derCE yvec outvec = (map (* (-1)) (zipWith zipAdd
                      (zipWith zipMult yvec (map (1/) outvec))
                      (zipWith zipMult (map (\x1 -> 1.0 - x1) yvec)
                            (map (1.0/) (map (\x -> 1.0 - x) outvec)))))

zipAdd :: Float -> Float -> Float
zipAdd a b = (a + b)

-- dervative of each output with respect to their input
-- takes in the input to the output nodes
derOut :: [Float] -> [Float]
derOut inp = (map (/(inpSum^2))
                (map (\x ->(exp x) *(inpSum - (exp x))) inp))
    where inpSum = (sum (map exp inp))

-- takes relu values
derRelu :: [Float] -> [Float]
derRelu inp = (map (\x -> if (x > 0) then 1 else 0) inp)


-- last layer to output--
-- derCE * derOut * layer before outvalue


-- backwards prop
-- TAKES A REVERSE MODEL/ERROR
-- takes the y of the output of the model and the expected y
-- TODO: I think the errors are in the wrong order
backwardsprop :: Model -> [Float]
                          -> [Float] -> Error -> IO Model
backwardsprop ((i, (b, w)):ms) out
  expout err@((beforeSM, afterSM):(beforeWeights, afterWeights):re) = do
      weight <- w
      let dce = (derCE expout out) --(vector)
      let dout = (derOut afterWeights) --(vector)
      let changeWeight = (map (* 0.10)
                            (zipWith3 zipMult2 dce dout beforeWeights))
      let newWeight = return (map (zipWith zipSub changeWeight) weight)
      otherMod <- (backprophelp ms dce dout weight err)
      -- new output layer weights
      return ((i, (b, newWeight)):otherMod)

-- layer to layer
-- derRelu * h1 outvalue * (derCE * derOut * (weights on 2nd layer output))

--input to layer
--derRelu * input value out * (previous relu * previous 3rd value * weight coming into layer)

-- TODO: a lot of this is repetitive, pull it out and create a helper
-- this is also good for testing, right now it is this way because of
-- dealing with the I/O

backprophelp :: Model -> [Float] ->  [Float]
                    -> [ [Float]] -> Error -> IO Model
backprophelp ((i, (b, w)):[]) d1 d2 weigh
  ((h2input, h2output):(h1input, h1output):re) = do
      currWeight <- w
      let thirdValue = (map (zipWith3 zipMult2 d1 d2) weigh) --[vector]
      let weightChange = (map (map (* 0.10))
                            (map (zipWith3 zipMult2
                                      (derRelu h2input) h1output) thirdValue))
      let newWeight = return (zipWith zipVecSub weightChange currWeight)
      return [(i, (b, newWeight))]

backprophelp ((i, (b, w)):ms) d1 d2 weigh
  ((h2input, h2output):(h1input, h1output):re) = do
      currWeight <- w
      let thirdValue = (map (zipWith3 zipMult2 d1 d2) weigh)
      let weightChange = (map (map (* 0.10))
                          (map (zipWith3 zipMult2
                              (derRelu h2input) h1output) thirdValue))
      let newWeight = return (zipWith zipVecSub weightChange currWeight)
      otherMod <- (backprophelp1 ms (derRelu h2input)
                    thirdValue currWeight ((h1input, h1output):re))
      return ((i, (b, newWeight)):otherMod)


backprophelp1 :: Model ->  [Float] -> [ [Float]]
                      -> [ [Float]] -> Error -> IO Model
backprophelp1 [] d1 d2 weigh _ = do
    return []

backprophelp1 ((i, (b, w)):[]) d1 d2 weigh _ = do
    return [(i, (b, w))]

backprophelp1 ((i, (b, w)):(i1, (b1, w1)):[]) d1 d2 weigh
  ((h2input, h2output):(h1input, h1output):re) = do
      currWeight <- w
      let maper = (map (zipWith zipMult d1) d2)
      let thirdValue = (zipWith zipVec maper weigh)
      let weightChange = (map (map (* 0.10))
                              (map (zipWith3 zipMult2
                                  (derRelu h2input) h1output) thirdValue))
      let newWeight = return (zipWith zipVecSub weightChange currWeight)
      return [(i, (b, newWeight))]

backprophelp1 ((i, (b, w)):ms) d1 d2 weigh
  ((h2input, h2output):(h1input, h1output):re) = do
      currWeight <- w
      let dRelu = (derRelu h2input)
      let maper = (map (zipWith zipMult d1) d2)
      let thirdValue = (zipWith zipVec maper weigh)
      let weightChange = (map (map (* 0.10))
                              (map (zipWith3 zipMult2
                                dRelu h1output) thirdValue))
      let newWeight = return (zipWith zipVecSub weightChange currWeight)
      otherMod <- (backprophelp1 ms dRelu
                    thirdValue currWeight ((h1input, h1output):re))
      return ((i, (b, newWeight)):otherMod)

--look to see where this is used and change it
zipVec ::  [Float] ->  [Float] ->  [Float]
zipVec a b = zipWith zipMult a b

zipVecSub ::  [Float] ->  [Float] ->  [Float]
zipVecSub a b = zipWith zipSub a b

zipMult2 :: Float -> Float -> Float -> Float
zipMult2 a b c= (a * b * c)

zipSub :: Float -> Float -> Float
zipSub a b = (b - a)
