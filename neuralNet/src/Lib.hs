module Lib where

import qualified Numeric.LinearAlgebra as NL
-- import qualified Data.Vector as DV
import System.Random
import Data.Functor
import Control.Monad
import Data.Ord
import Data.List
import Control.Parallel (par)
import Control.Parallel.Strategies (Strategy, parMap, parList, rseq, rdeepseq)


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
    let maxX = (maximum x) in 
        (map (/(summation)) toX)
        where 
            toX = (map exp (map (subtract maxX) x))
            summation = (sum (map exp (map (subtract maxX) x)))


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
type Model = [(Int, (IO [Float], IO[[Float]]))]

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
                            ((createB numOut), (createW numOut num))):wholemod))
                                , []) -- create the bias and weights for output layer
    (Hidden numHid) -> (((numHid,
                          ((createB numHid), (createW numHid num))):wholemod)
                                , lay)

-- possibly a more efficient way to do this--
-- but wanted to make sure it gives different ran numbers
-- I need to check that it does
createB :: Int -> IO[Float]
createB bin = return (replicate bin 1.0)
createW :: Int -> Int -> IO[[Float]]
createW num num2 = (replicateM num (createWHelp num2))

createWHelp :: Int -> IO [Float]
createWHelp num = (replicateM num (gauss (0.01)))


-- forward prop

forwardprop :: ([([Float], [[Float]])]) -> ([Float], Error)
                        -> ([Float], Error)
forwardprop [] (vec, err) = (vec, err)
forwardprop ((bias, weight):[]) (vec, err) =
    let output = (zipWith (+) bias
                    (map (foldl1 (+))
                    (multWeight (zipMultWeight vec
                    weight)))) in 
    ((softmax output), ([(output, (softmax output))] ++ err))

forwardprop ((bias, weight):ms) (vec, err) =
    let 
        output = (zipWith (+) bias
                    (map (foldl1 (+))
                    (multWeight (zipMultWeight vec
                        weight)))) 
        newerror = if (err == []) then ((output, (map relu output)):[(vec, vec)])
                    else ((output, (map relu output)):err) in 
    (forwardprop ms ((map relu output), newerror))

-- multiplying the first of every list together -- 
zipMultWeight :: [Float] -> [[Float]] -> [([Float],[Float])]
zipMultWeight a [] = []
zipMultWeight a (b:bs) = [(a,b)] ++ (zipMultWeight a bs)

multWeight [] = []
multWeight ((a,b):rs) = [(zipWith (*) a b)] ++ (multWeight rs)

zipMult :: Float -> Float -> Float
zipMult a b = (a * b)


-- backwards prop
--the derivative of cross entropy
-- takes in the y vector and the output vector
-- if statements so that it doesnt return nAn 
derCE :: [Float] -> [Float] -> [Float]
derCE yvec outvec = (map (* (-1)) (zipWith zipAdd
                      (zipWith zipMult yvec (map (1/) (map (\x3 -> if x3 == 0 then 0.0000001 else x3) outvec)))
                      (zipWith zipMult (map (\x1 -> 1.0 - x1) yvec)
                            (map (1.0/) (map (\x -> 1.0 - x) (map (\x2 -> if x2 == 1 then 1.0000001 else x2)outvec))))))

zipAdd :: Float -> Float -> Float
zipAdd a b = (a + b)

-- dervative of each output with respect to their input
-- takes in the input to the output nodes
-- CHECKED
derOut :: [Float] -> [Float]
derOut inp = (map (/(inpSum^2))
                (map (\x ->(exp x) *(inpSum - (exp x))) inp))
    where inpSum = (sum (map exp inp))

-- takes relu values
derRelu :: [Float] -> [Float]
derRelu inp = (map (\x -> if (x > 0) then 1 else 0) inp)


-- backwards prop
-- TAKES A REVERSE MODEL/ERROR
-- takes the y of the output of the model and the expected y

backwardsprop :: [([Float], [[Float]])] -> [Float]
                          -> [Float] -> Error -> [([Float], [[Float]])]
backwardsprop ((bias, weight):ms) output
  expout ((beforeSM, afterSM):(beforeRelu, afterRelu):re) =
      let 
        dce          = (derCE expout output)
        newbias      = (zipWith zipSub (map (*0.001) dce) bias)
        dout         = (derOut beforeSM) 
        changeWeight = (map (* 0.001)
                            (zipWith3 zipMult2 dce dout afterRelu))
        newWeight = (map (zipWith zipSub changeWeight) weight) in
      -- new output layer weights
      ((newbias, newWeight):(backprophelp ms dce dout weight ((beforeRelu, afterRelu):re)))

-- layer to layer
-- derRelu * h1 outvalue * (derCE * derOut * (weights on 2nd layer output))

--input to layer
--derRelu * input value out * (previous relu * previous 3rd value * weight coming into layer)

condense :: [[Float]] -> [Float]
condense (x:xs) = condenseHelp xs x

condenseHelp [] vec     = vec 
condenseHelp (m:ms) vec = condenseHelp ms (zipWith (+) m vec) 

backprophelp :: [([Float], [[Float]])] -> [Float] ->  [Float]
                    -> [ [Float]] -> Error -> [([Float], [[Float]])]

backprophelp ((bias, currWeight):[]) d1 d2 weigh
  ((h2input, h2output):(h1input, h1output):re) =
      let 
        newbias     = (zipWith zipSub (map (*0.002) d1) bias) 
        thirdValue  = (condense (map (zipWith (*) (zipWith (*) d1 d2)) weigh))
        weightChange = (map (* 0.002)
                            (zipWith3 zipMult2
                                      (derRelu h2input) h1output thirdValue))
        newWeight = (map (zipWith zipSub weightChange) currWeight) in
      [(newbias, newWeight)]

backprophelp ((bias, currWeight):ms) d1 d2 weigh
  ((h2input, h2output):(h1input, h1output):re) =
      let 
        newbias      = (zipWith zipSub (map (*0.002) d1) bias)  
        thirdValue   = (condense (map (zipWith3 zipMult2 d1 d2) weigh)) 
        weightChange = (map (* 0.002)
                          (zipWith3 (zipMult2)
                              (derRelu h2input) h1output thirdValue))
        newWeight    = (map (zipWith zipSub weightChange) currWeight)  in
        ((newbias, newWeight):(backprophelp ms (derRelu h2input)
                                thirdValue currWeight ((h1input, h1output):re)))


--look to see where this is used and change it
zipVec ::  [Float] ->  [Float] ->  [Float]
zipVec a b = zipWith zipMult a b

zipVecSub ::  [Float] ->  [Float] ->  [Float]
zipVecSub a b = zipWith zipSub a b

zipMult2 :: Float -> Float -> Float -> Float
zipMult2 a b c= (a * b * c)

zipSub :: Float -> Float -> Float
zipSub a b = (b - a)


learn x y layers = 
    let
        (vec, err) = (forwardprop layers (x, [])) in 
    (backwardsprop (reverse layers) vec y (reverse err))

feed vec layers = 
    let
        (vector, err) = (forwardprop layers (vec, [])) in
    vector
