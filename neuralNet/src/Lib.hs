module Lib where

import qualified Numeric.LinearAlgebra as NL
-- import qualified Data.Vector as DV
import System.Random
import Data.Functor
import Control.Monad
import Data.Ord
import Data.List
import Control.Parallel (par, pseq)
import Control.Parallel.Strategies (Strategy, parMap, parList, rseq, rdeepseq, using, evalList)


data Layer = Input Int
    | Output Int
    | Hidden Int

--Loss Functions

-- following https://deepnotes.io/softmax-crossentropy
--softmax

softmax :: [Double] ->  [Double]
softmax x =
    (parMap rdeepseq (/(summation)) toX)
    where 
        maxX = (maximum x)
        toX = (parMap rdeepseq exp (parMap rdeepseq (subtract maxX) x))
        summation = (sum (parMap rdeepseq exp (parMap rdeepseq (subtract maxX) x)))


--cross entropy loss
-- python code:
-- X is the output from fully connected layer (num_examples x num_classes)
-- y is labels (num_examples x 1)
--CHECK THIS becuase I think I have confused myself with x/y and where softmax
-- should be applied, I actually think on y now
cel :: [[Double]] -> [Double] -> [Double]
cel x y = parMap rdeepseq (\(a,b) -> a/ b )
  (zipWith zipIntTuple
    (parMap rdeepseq (sum)
      (parMap rdeepseq (parMap rdeepseq(\(yi,xi) -> -yi*(log xi))) zip1))
    len)
    where zip1 = (parMap rdeepseq (zipWith zipDoubleTuple y) (parMap rdeepseq softmax x))
          len  = (parMap rdeepseq (length) x)

-- telling how I want them zipped, idk what will happen if they are different size
-- so if I have issues come back and look at this
zipDoubleTuple :: Double -> Double -> (Double, Double)
zipDoubleTuple a b = (a , b)

zipIntTuple :: Double -> Int -> (Double, Double)
zipIntTuple a b = (a , (fromIntegral b))
-- activation functions --
-- relu, I didnt end up using this tho
relu :: Double -> Double
relu = max 0

-- log activation function that I used bc the partial derivate made more sense
logAct f1 = (1/(1+ exp(-f1))) 

type InputLayers = [Layer]
-- I need to keep track of the number nodes in the previous layer (which is why int is there)
-- There is probably a better way to do this
-- I end up converting the model to not have the int once it is created (see Main)
type Model = [(Int, (IO [Double], IO[[Double]]))]
-- need to keep track of weighted input and activation
type Error = [([Double], [Double])]


-- takes a standard dev as input
-- grabbed from the internet 
-- this is boxmuller
gauss :: Double -> IO Double
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
createB :: Int -> IO[Double]
createB bin = return (replicate bin 1.0)
createW :: Int -> Int -> IO[[Double]]
createW num num2 = (replicateM num (createWHelp num2))

createWHelp :: Int -> IO [Double]
createWHelp num = (replicateM num (gauss (0.01)))


-- forward prop
-- multiply the weight by the vector
-- each [] in weight represents all the weights for the input to
-- one node
forwardMult :: [Double] -> ([Double], [[Double]]) -> [Double]
forwardMult inputVector (bias, weights) = 
    (zipWith (+) bias
        (parMap rdeepseq (foldl1 (+))
        (zipMultWeight inputVector
            weights `using` parList rdeepseq)))
-- I wanted to use scanl here but could not figure out a way to calculate the error
-- without recalculating
forwardprop :: ([([Double], [[Double]])]) -> ([Double], Error)
                        -> ([Double], Error)
forwardprop (model:[]) (vec, err) =
    let output = forwardMult vec model  in 
    ((softmax output), ([(output, (softmax output))] ++ err))

forwardprop (model:ms) (vec, err) =
    let 
        output = forwardMult vec model
        newerror = if (err == []) then ((output, (parMap rdeepseq logAct output)):[(vec, vec)])
                    else ((output, (parMap rdeepseq logAct output)):err) in 
    (forwardprop ms ((parMap rdeepseq logAct output), newerror))
-- these all had zip in the name bc once upon a time I was using them with zip
-- but things spiraled and I havent changed the names
-- multiplying vector and matrix without transposing -- 
zipMultWeight :: [Double] -> [[Double]] -> [[Double]]
zipMultWeight a [] = []
zipMultWeight a (b:bs) = [(zipWith (*) a b)] ++ (zipMultWeight a bs)

--take a matrix and condense add together
zipAddWeight :: [[Double]] -> [Double]
zipAddWeight (a:as) = (zipAddWeighth a as) 

zipAddWeighth :: [Double] -> [[Double]] -> [Double]
zipAddWeighth a [] = a
zipAddWeighth a (b:bs) = (zipAddWeighth (zipWith (+) a b `using` parList rdeepseq) bs)

-- backwards prop
-- TAKES A REVERSE MODEL

-- this does the math that is located here https://www.ics.uci.edu/~pjsadows/notes.pdf

-- takes the y of the output of the model and the expected y
-- takes in vector of input 
--version2 not reusing stuff bc debugging but realistically v1000000
-- this is from the output layer to the first hidden layer
-- (output - expected output) * hiOutput (input to previous layer after activation)
backwardspropV2 :: [([Double], [[Double]])] -> Double
                        -> [Double] -> Error -> [([Double], [[Double]])]
backwardspropV2 ((bias, weight):ms) learnRate
  expout er@((beforeSM, output):(hiInput, hiOutput):re) =
    --calculate total error (actual - expected)
    let
        totalError  = zipWith (-) output expout
        --previous output * current weights 
        multW       = multHelp hiOutput totalError 
        -- multiply by learning rate
        learnWeight = parMap rdeepseq (parMap rdeepseq (*learnRate)) multW
        --subtract from old weights
        newWeights  = weightMatrix weight (transpose learnWeight)
        --new bias -- 
        newBias     = (zipWith (-) bias (parMap rdeepseq (*learnRate) totalError)) in
    (newBias, newWeights):(bpV2Helper ms learnRate totalError weight ((hiInput, hiOutput):re))

multHelp [] err = []
multHelp (h:hout) err = (map (*h) err):(multHelp hout err)

--matrix mult --
mm2 [] [] = []
mm2 (m:m1) (n:n2) = (zipWith (*) m n):(mm2 m1 n2)

-- for each input in the input vector multiply against loss matrix
--  and then add down to a vetor creating a loss vector with respect to input

--should this be zipAddWeight or should it be fold
someBackPropMath :: [Double] -> [[Double]] -> [[Double]]
someBackPropMath [] _        = []
someBackPropMath (x:xs) matr = (zipAddWeight (parMap rdeepseq (map (* x)) matr)):(someBackPropMath xs matr)

bpV2Helper ((bias, weight):[]) learnRate
  totErr prevWeight ((hiInput, hiOutput):(input, _):[]) =
    let
        -- hiOutput (1- hiOutput)
        weightedOut    = zipWith (*) hiOutput (map (1-) hiOutput)
        -- weightOut * totalErr should return a maxtrix the same dimensions as weights
        multErr        = (multHelp totErr weightedOut)
        --multErr summed down for bias
        multBias      = zipAddWeight multErr
        -- multiply two matrix same dimensions without transpose
        prevWeighErr   = mm2 multErr prevWeight
        -- multiply each input by the weightErr and then sum those down the columns

        -- do this for each input node
        step3          = someBackPropMath input prevWeighErr 
        -- mult by learnRate --
        learnWeight    =  parMap rdeepseq (map (* learnRate)) step3
        -- update weight
        newWeight      = weightMatrix weight (transpose learnWeight)

        newBias        = (zipWith (-) bias (parMap rdeepseq (*learnRate) multBias) ) in
    [(newBias, newWeight)]  

    --modify this so for other layers --
-- bpV2Helper ((bias, weight):ms) learnRate
--     totErr prevWeight ((hiInput, hiOutput):(beforeAct, input):rs) =
--       let
--           -- hiOutput (1- hiOutput)
--           weightedOut    = zipWith (*) hiOutput (map (1-) hiOutput)
--           -- weightOut * totalErr should return a maxtrix the same dimensions as weights
--           multErr        = (multHelp totErr weightedOut)
--           --multErr summed down for bias
--           multBias      = zipAddWeight multErr
--           -- multiply two matrix same dimensions without transpose
--           prevWeighErr   = mm2 multErr prevWeight
--           -- multiply each input by the weightErr and then sum those down the columns
  
--           -- do this for each input node
--           step3          = someBackPropMath input prevWeighErr 
--           -- mult by learnRate --
--           learnWeight    =  parMap rdeepseq (map (* learnRate)) step3
--           -- update weight
--           newWeight      = weightMatrix weight (transpose learnWeight)
  
--           newBias        = (zipWith (+) bias (parMap rdeepseq (*learnRate) multBias) ) in
--       (newBias, newWeight):(bpV2Helper ms learnrate totErr weight (beforeAct, input):rs)  
  
--     (newBias, newWeight):(bpV2Helper ms learnRate addWeightToVec prevWeight ((hjInput, hjOutput):rs))

weightMatrix [] []             = []
weightMatrix (m1:rm1) (m2:rm2) = (zipWith (-) m1 m2):(weightMatrix rm1 rm2)


-- zipMultWeightErr [] []     = []                         
-- zipMultWeightErr (a:ra) (b:rb) =  (zipWith (*) a b):(zipMultWeightErr ra rb)

-- functions written to make my code compatible with 
-- the main function I grabbed off the internet
learn x y layers = 
    let
        (vec, err) = (forwardprop layers (x, [])) in 
    (reverse (backwardspropV2 (reverse layers) (0.01) y err))


feed vec layers = 
    let
        (vector, err) = (forwardprop layers (vec, [])) in
    vector
